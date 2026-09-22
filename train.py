"""Portable Qwen2.5-VL LoRA SFT with paper hyperparameters.

This cleaned implementation is not a recovered historical experiment checkpoint.
"""
import argparse
import json
import os
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['grounding','relation','navigation'],required=True)
    p.add_argument('--data',type=Path,required=True,help='JSONL with instruction, output and images')
    p.add_argument('--image-root',type=Path,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--previous-adapter',help='Previous stage adapter to merge before initializing this stage')
    p.add_argument('--max-pixels',type=int,default=1024*28*28)
    p.add_argument('--max-length',type=int,default=8192,help='Fail on overflow; never silently truncate image tokens')
    p.add_argument('--seed',type=int,default=42)
    return p.parse_args()


def messages(row, root):
    content = []
    for value in row['images']:
        path = (root/value).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f'Invalid image path: {value}')
        content.append({'type':'image','image':str(path)})
    if not content or not row['instruction'].strip() or not row['output'].strip():
        raise ValueError('Empty image, instruction or target')
    content.append({'type':'text','text':row['instruction']})
    return [{'role':'user','content':content}]


class SupervisedCollator:
    """Process one image example at a time; global batch size comes from accumulation."""
    def __init__(self,processor,root,max_length):
        self.processor,self.root,self.max_length=processor,root,max_length

    def __call__(self,rows):
        import torch
        from qwen_vl_utils import process_vision_info
        if len(rows)!=1:
            raise ValueError('Use microbatch size 1 for variable-resolution image tokens')
        row=rows[0]
        user=messages(row,self.root)
        full=user+[{'role':'assistant','content':[{'type':'text','text':row['output']}]}]
        images,videos=process_vision_info(user)
        prompt_text=self.processor.apply_chat_template(user,tokenize=False,add_generation_prompt=True)
        full_text=self.processor.apply_chat_template(full,tokenize=False,add_generation_prompt=False)
        prompt=self.processor(text=[prompt_text],images=images,videos=videos,return_tensors='pt')
        batch=self.processor(text=[full_text],images=images,videos=videos,return_tensors='pt')
        length=prompt['input_ids'].shape[1]
        if batch['input_ids'].shape[1]>self.max_length:
            raise ValueError(f"Sample {row.get('id','?')} exceeds --max-length; increase it or reduce --max-pixels")
        if not torch.equal(prompt['input_ids'],batch['input_ids'][:,:length]):
            raise ValueError('Chat template prompt is not a prefix of supervised text')
        labels=batch['input_ids'].clone()
        labels[:,:length]=-100
        labels[batch['attention_mask']==0]=-100
        if not (labels!=-100).any():
            raise ValueError('No supervised answer tokens')
        batch['labels']=labels
        return batch


def main():
    args=arguments()
    import torch
    from datasets import Dataset
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,Trainer,TrainingArguments,set_seed
    from peft import LoraConfig,TaskType,get_peft_model,PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU')
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world>8 or 8%world:
        raise ValueError('Use 1, 2, 4 or 8 processes to retain effective batch size 8')
    rows=[json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Training dataset is empty')
    for row in rows:
        messages(row,args.image_root)
    set_seed(args.seed)
    nav=args.stage=='navigation'
    settings={'learning_rate':2e-5 if nav else 1e-4,'epochs':2 if nav else 1,
              'lora_r':128 if nav else 32,'lora_alpha':256 if nav else 16}
    bf16=torch.cuda.is_bf16_supported()
    dtype=torch.bfloat16 if bf16 else torch.float16
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=256*28*28,max_pixels=args.max_pixels)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,torch_dtype=dtype,attn_implementation='sdpa')
    if args.previous_adapter:
        model=PeftModel.from_pretrained(model,args.previous_adapter).merge_and_unload()
    # Restrict adapters to language-model projections, keeping the visual encoder frozen.
    targets=[name for name,_ in model.named_modules() if 'visual' not in name and name.rsplit('.',1)[-1] in
             {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}]
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=settings['lora_r'],
                         lora_alpha=settings['lora_alpha'],lora_dropout=0.05,target_modules=targets,bias='none'))
    model.config.use_cache=False
    model.enable_input_require_grads()
    training_args=TrainingArguments(output_dir=args.output,per_device_train_batch_size=1,
        gradient_accumulation_steps=8//world,num_train_epochs=settings['epochs'],
        learning_rate=settings['learning_rate'],optim='adamw_torch',bf16=bf16,fp16=not bf16,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        remove_unused_columns=False,report_to='none',save_strategy='epoch',logging_steps=10,
        seed=args.seed,label_names=['labels'],ddp_find_unused_parameters=False)
    trainer=Trainer(model=model,args=training_args,train_dataset=Dataset.from_list(rows),
                    data_collator=SupervisedCollator(processor,args.image_root,args.max_length))
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output)
        config={**vars(args),**settings,'effective_batch_size':8,'samples':len(rows)}
        Path(args.output,'run_config.json').write_text(json.dumps(config,default=str,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
