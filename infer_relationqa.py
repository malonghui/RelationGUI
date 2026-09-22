"""Generate RelationQA predictions using the marked screenshot and crop."""
import argparse
import json
from pathlib import Path
from evaluate_relationqa import load,LABELS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--adapters',nargs='*',default=[],help='Stage adapters in order; each merged before the next')
    p.add_argument('--max-pixels',type=int,default=1024*28*28)
    args=p.parse_args()
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,torch_dtype=dtype,device_map='auto')
    for adapter in args.adapters:
        model=PeftModel.from_pretrained(model,adapter).merge_and_unload()
    model.eval()
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=256*28*28,max_pixels=args.max_pixels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:
        for row in load(args.data_root/'annotations.jsonl'):
            content=[]
            for kind in ('marked','crop'):
                path=(args.data_root/row['images'][kind]).resolve()
                if not path.is_relative_to(args.data_root.resolve()) or not path.is_file():
                    raise ValueError('Invalid image path')
                content.append({'type':'image','image':str(path)})
            prompt=row['question']+'\nOptions:\n'+'\n'.join(row['options'])+'\nReturn exactly one label: '+', '.join(LABELS)+'.'
            content.append({'type':'text','text':prompt})
            messages=[{'role':'user','content':content}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            images,videos=process_vision_info(messages)
            inputs=processor(text=[text],images=images,videos=videos,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                ids=model.generate(**inputs,max_new_tokens=16,do_sample=False)
            answer=processor.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
            out.write(json.dumps({'id':row['id'],'prediction':answer},ensure_ascii=False)+'\n')
            out.flush()


if __name__=='__main__':
    main()
