"""Evaluate RelationQA top-1 exact-match accuracy on the complete supplied benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path

LABELS=('trigger','complement','parallel','none')


def load(path):
    rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    if len({x['id'] for x in rows})!=len(rows):
        raise ValueError('Duplicate IDs in '+str(path))
    return rows


def score(gold,predictions):
    gold_ids={x['id'] for x in gold}
    if not gold:
        raise ValueError('Empty benchmark')
    if len(gold_ids)!=len(gold) or len({x['id'] for x in predictions})!=len(predictions):
        raise ValueError('Duplicate IDs')
    pred={x['id']:x['prediction'] for x in predictions}
    if set(pred)-gold_ids:
        raise ValueError('Unknown prediction IDs')
    totals=Counter(); correct=Counter(); missing=invalid=0
    for row in gold:
        if row['answer'] not in LABELS:
            raise ValueError('Invalid gold answer')
        value=pred.get(row['id'])
        missing+=row['id'] not in pred
        invalid+=row['id'] in pred and value not in LABELS
        for key in ['overall','platform/'+row['platform'],'label/'+row['answer']]:
            totals[key]+=1
            correct[key]+=value==row['answer']
    return {'metric':'top-1 exact-match accuracy','missing':missing,'invalid':invalid,
            'coverage':len(pred)/len(gold),
            'scores':{k:{'correct':correct[k],'total':v,'accuracy':correct[k]/v} for k,v in sorted(totals.items())}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(score(load(args.annotations),load(args.predictions)),indent=2))


if __name__=='__main__':
    main()
