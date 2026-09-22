"""Build a traceable, quality-filtered release from the author's source tree.

This rebuild is NOT asserted to be the original experimental training split.
Only allowlisted JSON fields and referenced images enter the output directory.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

from PIL import Image

LABELS = ('trigger', 'complement', 'parallel', 'none')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def relative_source(value):
    value = str(value).replace('\\', '/')
    if '/all/' in value:
        value = value.rsplit('/all/', 1)[1]
    if value.startswith('/') or ':' in value or '..' in Path(value).parts:
        raise ValueError('Unrecognized source path')
    return value


def canonical(value):
    value = relative_source(value)
    value = value.replace('/data_low_', '/data_')
    value = re.sub(r'_(?:line_)?cropped_\d+(?=\.)', '', value)
    return re.sub(r'(?:_low|-low)(?=\.)', '', value)


def valid_box(points, w, h):
    if not isinstance(points, list) or len(points) != 2:
        return False
    try:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return all(math.isfinite(float(v)) for v in (x1,y1,x2,y2)) and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    except (TypeError, ValueError):
        return False


def source_files(root, platform):
    base = root/platform
    pattern = '*_relation.json' if platform == 'mobile' else '*.json'
    for path in sorted(base.rglob(pattern)):
        if 'finetune' not in path.name and 'data_low_' not in str(path):
            obj = read(path)
            if isinstance(obj, dict) and isinstance(obj.get('relationships'), list):
                yield path, obj


def build(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty; source files are never overwritten')
    root = source/'data/dataset/all'
    qa_root = source/'data/benchmark/relationqa'
    qa_raw = read(qa_root/'qa_benchmark.json')
    qa_sources = {canonical(x['pre_image']) for x in qa_raw}
    report = {'format_version':'2026.09-cleaned',
              'historical_split_reproduced':False,
              'paper':{'desktop':6854,'mobile':20284,'web':10586,'relationqa':1009},
              'policy': ['use actual image dimensions', 'require complete in-bounds boxes',
                         'require unambiguous contiguous group-to-relation alignment',
                         'exclude all RelationQA source screenshots and decoded-pixel duplicates from training',
                         'deduplicate normalized records', 'preserve original relation descriptions and labels'],
              'platforms':{}, 'benchmark':{}, 'source_files':[]}
    cache = {}

    def inspect(path):
        path = path.resolve()
        if path not in cache:
            with Image.open(path) as image:
                image.load()
                rgb = image.convert('RGB')
                pixel_hash = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                cache[path] = (image.size, pixel_hash)
        return cache[path]

    def copy_image(src, dest, archive):
        target = output/archive/dest
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    qa_hashes = set()
    for value in sorted(qa_sources):
        _, h = inspect(root/value)
        qa_hashes.add(h)
    rejects = []
    qa_rows = []
    qa_seen = set()
    for i, row in enumerate(qa_raw):
        if row.get('answer') not in LABELS or not row.get('question') or len(row.get('options', [])) != 4:
            raise ValueError(f'Invalid RelationQA annotation at index {i}')
        source_rel = canonical(row['pre_image'])
        files = {'original':root/source_rel}
        for name, field in [('marked','line_img_path'),('crop','crop_img_path')]:
            rel = row[field].replace('\\','/')
            if not rel.startswith('qa_benchmark_images/') or '..' in Path(rel).parts:
                raise ValueError(f'Unsafe benchmark image at {i}')
            files[name] = qa_root/rel
        images = {}
        for name, path in files.items():
            inspect(path)
            dest = Path('images')/(f'{i:04d}_{name}'+path.suffix.lower())
            copy_image(path, dest, 'RelationQA')
            images[name] = dest.as_posix()
        key = digest([source_rel,row['question'],row['options'],row['answer']])
        if key in qa_seen:
            raise ValueError('Duplicate benchmark item; historical count needs review')
        qa_seen.add(key)
        qa_rows.append({'id':'relationqa-'+key[:24], 'source_id':digest(source_rel),
                        'platform':row['platform'], 'question':row['question'],
                        'options':row['options'], 'answer':row['answer'], 'images':images})
    jsonl(output/'RelationQA/annotations.jsonl',qa_rows)
    report['benchmark'] = {'samples':len(qa_rows),'source_screenshots':len(qa_sources),
                           'platforms':dict(Counter(x['platform'] for x in qa_rows)),
                           'labels':dict(Counter(x['answer'] for x in qa_rows))}
    jsonl(output/'RelationQA/source_exclusions.jsonl', [{'source_id':digest(x),'source':x} for x in sorted(qa_sources)])
    all_relations, grounding, referring, train_relations = [], [], [], []
    relation_seen, element_seen = set(), set()
    for platform in ('desktop','mobile','web'):
        counts = Counter()
        for path, obj in source_files(root,platform):
            relpath = path.relative_to(root).as_posix()
            report['source_files'].append({'path':relpath, 'sha256':sha256(path)})
            rels, shapes = obj['relationships'], obj.get('shapes',[])
            counts['raw_screenshots'] += 1
            counts['raw_relation_groups'] += len(rels)
            image = path.with_name(path.stem.removesuffix('_relation')+('.jpg' if platform=='mobile' else '.png'))
            try:
                (w,h), pixel_hash = inspect(image)
            except (OSError, ValueError):
                rejects.append({'source':relpath,'reason':'missing_or_corrupt_image','relations':len(rels)})
                counts['rejected_missing_or_corrupt_image'] += len(rels)
                continue
            if (w,h) != (obj.get('imageWidth'),obj.get('imageHeight')):
                counts['corrected_dimension_metadata'] += 1
            group_order = []
            grouped = defaultdict(list)
            for s in shapes:
                gid = str(s.get('group_id'))
                if not group_order or group_order[-1] != gid:
                    group_order.append(gid)
                grouped[gid].append(s)
            if len(group_order)!=len(rels) or len(set(group_order))!=len(group_order):
                rejects.append({'source':relpath,'reason':'ambiguous_group_alignment','relations':len(rels)})
                counts['rejected_ambiguous_group_alignment'] += len(rels)
                continue
            source_rel = canonical(image.relative_to(root).as_posix())
            heldout = source_rel in qa_sources or pixel_hash in qa_hashes
            dest = Path('images')/image.relative_to(root)
            for index,(gid,rel) in enumerate(zip(group_order,rels)):
                elements = grouped[gid]
                reason = None
                if len(elements)<2:
                    reason = 'fewer_than_two_elements'
                elif any(not valid_box(s.get('points'),w,h) for s in elements):
                    reason = 'invalid_or_out_of_bounds_box'
                elif not isinstance(rel,dict) or not isinstance(rel.get('function_rel'),str) or not rel['function_rel'].strip():
                    reason = 'missing_function_description'
                elif any(not str(s.get('caption','')).strip() for s in elements):
                    reason = 'missing_element_caption'
                if reason:
                    rejects.append({'source':relpath,'group_index':index,'reason':reason})
                    counts['rejected_'+reason] += 1
                    continue
                clean = []
                for s in elements:
                    x1,y1 = s['points'][0]; x2,y2 = s['points'][1]
                    clean.append({'type':s.get('label',''), 'caption':s['caption'],
                                  'operation_result':s.get('result',''),
                                  'bbox':[x1/w,y1/h,x2/w,y2/h],
                                  'hierarchy':s.get('group_information','')})
                relation = {k:rel[k] for k in ('position_rel','layer_rel','function_rel','position','layer','function') if k in rel}
                key = digest([pixel_hash,clean,relation])
                if key in relation_seen:
                    counts['removed_exact_duplicates'] += 1
                    rejects.append({'source':relpath,'group_index':index,'reason':'exact_duplicate'})
                    continue
                relation_seen.add(key)
                row = {'id':'relation-'+key[:24], 'source_id':digest(source_rel),
                       'source_file':relpath,'source_group_index':index,
                       'image':dest.as_posix(),'image_sha256':sha256(image),
                       'pixel_sha256':pixel_hash,'width':w,'height':h,'platform':platform,
                       'split':'benchmark_source' if heldout else 'train',
                       'elements':clean, 'relation':relation}
                all_relations.append(row)
                counts['accepted_relation_groups'] += 1
                if heldout:
                    counts['excluded_benchmark_source_groups'] += 1
                else:
                    counts['training_relation_groups'] += 1
                    for dimension, field in [('functional','function_rel'),('spatial','position_rel'),('hierarchical','layer_rel')]:
                        answer = relation.get(field)
                        if answer:
                            answer = answer if isinstance(answer,str) else json.dumps(answer,ensure_ascii=False)
                            prompt = 'Describe the '+dimension+' relationship among these GUI elements (bounding boxes normalized to [0, 1]):\n'+json.dumps(clean,ensure_ascii=False)
                            train_relations.append({'id':row['id']+'-'+dimension,'source_id':row['source_id'],
                                                    'images':[row['image']],'instruction':prompt,'output':answer})
                    for e in clean:
                        ek = digest([pixel_hash,e])
                        if ek in element_seen:
                            continue
                        element_seen.add(ek)
                        common = {'id':'element-'+ek[:24],'source_id':row['source_id'],'images':[row['image']]}
                        grounding.append({**common,'instruction':'Locate the GUI element: '+e['caption']+'. Return [x1, y1, x2, y2] normalized to [0, 1].',
                                          'output':json.dumps([round(v,6) for v in e['bbox']])})
                        referring.append({**common,'instruction':'Describe the GUI element at normalized bounding box '+json.dumps(e['bbox'])+'.',
                                          'output':e['caption']})
                copy_image(image,dest,'RelationGUI')
            if counts['raw_screenshots'] % 500 == 0:
                print(platform,counts['raw_screenshots'],'screens checked',flush=True)
        report['platforms'][platform] = dict(counts)
        print(platform,dict(counts),flush=True)
    jsonl(output/'RelationGUI/annotations.jsonl',all_relations)
    for name,rows in [('grounding',grounding),('referring',referring),('relation',train_relations)]:
        jsonl(output/f'RelationGUI/train/{name}.jsonl',rows)
    report['training_rows'] = {'grounding':len(grounding),'referring':len(referring),'relation':len(train_relations)}
    report['accepted_relation_groups'] = len(all_relations)
    report['exclusion_counts'] = dict(Counter(x['reason'] for x in rejects))
    report['qualification'] = ('Quality-filtered rebuild from surviving source annotations, not a recovered historical 37,724-sample release. '
                               'Desktop crops and legacy synthetic negatives are omitted because the original inclusion list is unverified. '
                               'Benchmark-source records in annotations.jsonl are documentation only and excluded from every train file. '
                               'No new human semantic review or model-result reproduction is claimed.')
    jsonl(output/'RelationGUI/audit/exclusions.jsonl',rejects)
    dump(output/'RelationGUI/audit/report.json',report)
    dump(output/'RelationQA/audit.json',report['benchmark'])
    dump(output/'audit_report.json',report)
    return report


def pack(output):
    output = output.resolve()
    checksums = []
    for name in ('RelationGUI','RelationQA'):
        path = output/(name+'-cleaned.zip')
        temporary = path.with_suffix('.zip.partial')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
            for f in sorted((output/name).rglob('*')):
                if f.is_file():
                    z.write(f,f.relative_to(output).as_posix())
        with zipfile.ZipFile(temporary) as z:
            failure = z.testzip()
            if failure:
                raise ValueError('Corrupt archive member: '+failure)
        temporary.replace(path)
        checksums.append(sha256(path)+'  '+path.name)
        print(path.name,path.stat().st_size,flush=True)
    (output/'SHA256SUMS').write_text('\n'.join(checksums)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pack-only',action='store_true')
    parser.add_argument('--pack',action='store_true')
    args = parser.parse_args()
    if not args.pack_only:
        if not args.source:
            parser.error('--source is required to build')
        build(args.source,args.output)
    if args.pack or args.pack_only:
        pack(args.output)


if __name__ == '__main__':
    main()
