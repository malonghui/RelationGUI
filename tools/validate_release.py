"""Validate IDs, image integrity, coordinates and source-level split isolation."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import zipfile
from PIL import Image


def read_jsonl(path):
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len({x['id'] for x in rows})!=len(rows):
        raise ValueError('Duplicate IDs: '+str(path))
    return rows


@lru_cache(maxsize=None)
def resolve_image(root,value):
    p=(root/value).resolve()
    if Path(value).is_absolute() or not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError('Unsafe or missing image: '+value)
    return p


def validate(root):
    gui=root/'RelationGUI'; qa=root/'RelationQA'
    relations=read_jsonl(gui/'annotations.jsonl')
    benchmark=read_jsonl(qa/'annotations.jsonl')
    heldout_sources={json.loads(x)['source_id'] for x in (qa/'source_exclusions.jsonl').read_text(encoding='utf-8').splitlines()}
    if len(benchmark)!=1009:
        raise ValueError('RelationQA must retain the 1,009 original questions')
    images=set(); benchmark_pixels=set(); decoded_files=set()
    for row in benchmark:
        if row['answer'] not in {'trigger','complement','parallel','none'}:
            raise ValueError('Unexpected QA label')
        for value in row['images'].values():
            images.add(resolve_image(qa,value))
        original=resolve_image(qa,row['images']['original'])
        with original.open('rb') as f:
            file_hash=hashlib.file_digest(f,'sha256').hexdigest()
        if file_hash not in decoded_files:
            with Image.open(original) as im:
                rgb=im.convert('RGB')
                benchmark_pixels.add(hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest())
            decoded_files.add(file_hash)
    train_sources=set(); train_pixels=set()
    for row in relations:
        images.add(resolve_image(gui,row['image']))
        if row['split']=='train':
            train_sources.add(row['source_id']); train_pixels.add(row['pixel_sha256'])
        for e in row['elements']:
            x1,y1,x2,y2=e['bbox']
            if not 0<=x1<x2<=1 or not 0<=y1<y2<=1:
                raise ValueError('Invalid normalized box')
    if train_sources & heldout_sources or train_pixels & benchmark_pixels:
        raise ValueError('Training/benchmark source overlap')
    train_rows={}
    for task in ('grounding','referring','relation'):
        rows=read_jsonl(gui/f'train/{task}.jsonl')
        train_rows[task]=len(rows)
        for row in rows:
            if row['source_id'] not in train_sources or row['source_id'] in heldout_sources:
                raise ValueError('Training row references an excluded source')
            if not row['instruction'].strip() or not row['output'].strip():
                raise ValueError('Empty training instruction/answer')
            for value in row['images']:
                images.add(resolve_image(gui,value))
    for index,path in enumerate(sorted(images)):
        with Image.open(path) as image:
            image.verify()
        if index and index%2000==0:
            print(index,'images verified',flush=True)
    for archive in root.glob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).is_absolute() or '..' in Path(name).parts or Path(name).suffix.lower() not in {'.json','.jsonl','.png','.jpg','.jpeg','.md','.txt'}:
                    raise ValueError('Unexpected archive member: '+name)
            if z.testzip():
                raise ValueError('Corrupt ZIP archive')
    sums=root/'SHA256SUMS'
    if sums.exists():
        for line in sums.read_text().splitlines():
            expected,name=line.split('  ',1)
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Unsafe checksum path')
            with path.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=expected:
                    raise ValueError('Archive SHA256 mismatch')
    result={'relation_groups':len(relations),'training_rows':train_rows,'relationqa':len(benchmark),
            'verified_images':len(images),'training_benchmark_source_overlap':0}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    validate(p.parse_args().root)
