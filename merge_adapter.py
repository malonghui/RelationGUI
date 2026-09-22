"""Merge stage adapters in training order into a standalone checkpoint."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--adapters',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.float32,device_map='cpu')
    for adapter in a.adapters:
        model=PeftModel.from_pretrained(model,adapter).merge_and_unload()
    model.save_pretrained(a.output)
    AutoProcessor.from_pretrained(a.model).save_pretrained(a.output)

if __name__=='__main__':
    main()
