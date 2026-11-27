"""Fit action-specific word sequences from the command outside the annotated target clause."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ngrams(text):
    # Never form a feature across a sentence boundary.
    result=set()
    for sentence in re.split(r'[.!?;:]',unicodedata.normalize('NFKC',text).casefold()):
        words=re.findall(r"[^\W\d_]+|\d+",sentence)
        for length in (1,2,3):
            result.update(' '.join(words[j:j+length]) for j in range(len(words)-length+1))
    return result


def fit(manifests,release,out):
    path=manifests/'fit.jsonl'
    spec=json.loads((manifests/'split-lock.json').read_bytes())['splits']['fit']
    if digest(path)!=spec['sha256']:raise ValueError('Wrong fit split')
    fit_rows=list(map(json.loads,path.open()))
    if len(fit_rows)!=spec['count'] or len({r['source_group'] for r in fit_rows})!=len(fit_rows):
        raise ValueError('Expected a distinct fit photo per row')
    originals={v['id']:v for v in map(json.loads,release.open())}
    all_counts=defaultdict(Counter)
    support=defaultdict(Counter)
    labels=defaultdict(set)
    used=Counter()
    for row in fit_rows:
        action=row.get('operation')
        if not action:continue
        text=row['instruction']
        for word in ngrams(text):
            all_counts[word][action]+=1
        orig=originals[row['id']]
        if orig['instruction']!=text or orig['sha256']['source.png']!=row['source_sha256'] or orig['operation']!=action:
            raise ValueError('Fit row differs from release')
        knowledge=orig.get('knowledge',{})
        clause=knowledge.get('clue')
        if not clause or text.count(clause)!=1:continue
        before,after=text.split(clause)
        premise=knowledge.get('scenario')
        if premise:
            if not before.startswith(premise):continue
            before=before[len(premise):]
        # Tie-breaks describe which instance to edit, not the requested operation.
        after=re.split(r'(?<=[.!?])\s+',after,maxsplit=1)[0]
        vocab=ngrams(before) | ngrams(after)
        label=orig['target_annotation']['label'].casefold()
        for word in vocab:
            support[word][action]+=1
            labels[word].add(label)
        used[action]+=1
    terms={}
    for word,counts in support.items():
        # A retained phrase has also never occurred in the full instructions of a different
        # action, including the target clauses excluded from the positive evidence.
        if len(counts)!=1 or len(all_counts[word])!=1 or len(labels[word])<6:continue
        (action,n),=counts.items()
        if n>=12:
            terms[word]=dict(action=action,fit_photos=n,target_classes=len(labels[word]))
    if set(used)!={v['action'] for v in terms.values()}:raise ValueError('Missing an action')
    report=dict(fit=digest(path),released_pairs=digest(release),training_code=digest(Path(__file__)),
                min_photos=12,min_classes=6,positive_counts=used,terms=terms)
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('fitted',sum(used.values()),'photos;',len(terms),'supported phrases')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--release-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    fit(args.manifests,args.release_manifest,args.output)
