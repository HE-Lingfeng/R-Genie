"""Create a development partition and supervision from the merged *training* manifest."""
import argparse
import fcntl
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata


SALT = 'REditBench-merged/latent-intention-development/v1/'
GEOMETRIC = {'move', 'enlarge', 'shrink'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def stable(key):
    return digest((SALT+key).encode())


def output_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True)+'\n').encode()


def assign(rows, count):
    repeats = Counter(norm(r['instruction']) for r in rows)
    buckets = defaultdict(list)
    for r in rows:
        if (r['origin'].startswith('original') or r['instruction_type']!='implicit'
                or repeats[norm(r['instruction'])]!=1):
            continue
        buckets[r['origin'],r['operation']].append(r)
    eligible = sum(map(len,buckets.values()))
    if not 0<count<eligible:
        raise ValueError('Invalid development count relative to eligible sources')
    quotas = {k: len(v)*count//eligible for k,v in buckets.items()}
    remainder = sorted(buckets,key=lambda k:(-(len(buckets[k])*count%eligible),k))
    for k in remainder[:count-sum(quotas.values())]:
        quotas[k]+=1
    development = set()
    for k,v in buckets.items():
        chosen=sorted(v,key=lambda r:stable(r['source_group']))[:quotas[k]]
        development.update(r['id'] for r in chosen)
    if len(development)!=count:
        raise ValueError('Duplicate assignment')
    return development


def supervision(row, expanded):
    item=dict(row)
    if row['origin'].startswith('original'):
        item['supervision']='paired-change-only'
        return item
    original=expanded[row['id']]
    if (original['instruction']!=row['instruction'] or original['operation']!=row['operation']
        or original['sha256'][Path(original['source_image']).name]!=row['source_sha256']
        or original['sha256'][Path(original['edited_image']).name]!=row['target_sha256']):
        raise ValueError(f"Construction provenance mismatch for {row['id']}")
    for field in ('target_mask','edit_region','after_mask'):
        if field in original:
            expected=original['sha256'][Path(original[field]).name]
            if row[field+'_sha256']!=expected:
                raise ValueError(f'Mismatched {field} for '+row['id'])
    item['supervision']='native-instance-and-edit-region'
    item['canvas']=original['canvas']
    if row['operation'] in GEOMETRIC:
        params=original['params']
        w,h=item['canvas']
        x0,y0,x1,y1=params['dst_box']
        if min(w,h)<=0 or not (0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError('Invalid destination geometry '+row['id'])
        geo={'destination_box_xyxy':[x0/w,y0/h,x1/w,y1/h]}
        if row['operation']=='move':
            geo.update(dx=params['dx']/w,dy=params['dy']/h)
        else:
            geo['scale']=params['factor']
        item['geometry']=geo
    return item


def write_stable(path, content):
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('Existing partition has different contents: '+str(path))
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(content)
        temp=Path(f.name)
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged',required=True,type=Path)
    parser.add_argument('--expanded',required=True,type=Path,help='Released construction manifest JSONL')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--development-size',type=int,default=500)
    a=parser.parse_args()
    seal=json.loads((a.merged/'split-lock.json').read_bytes())
    raw=(a.merged/'train.jsonl').read_bytes()
    if digest(raw)!=seal['manifests']['train.jsonl']:raise ValueError('Training manifest has changed')
    rows=list(map(json.loads,raw.splitlines()))
    for field in ('id','source_group','source_sha256','target_sha256'):
        if len({r[field] for r in rows})!=len(rows):raise ValueError('Training manifest repeats '+field)
    needed={r['id'] for r in rows if not r['origin'].startswith('original')}
    expanded={}
    with a.expanded.open() as f:
        for line in f:
            r=json.loads(line)
            if r['id'] in needed:
                if r['id'] in expanded:raise ValueError('Duplicate expansion ID '+r['id'])
                expanded[r['id']]=r
    if set(expanded)!=needed:raise ValueError('Missing construction records')
    development=assign(rows,a.development_size)
    sets={'fit':[],'development':[]}
    for r in rows:
        sets['development' if r['id'] in development else 'fit'].append(supervision(r,expanded))
    for field in ('source_group','source_sha256','target_sha256'):
        if {r[field] for r in sets['fit']} & {r[field] for r in sets['development']}:
            raise ValueError('Overlap across partitions: '+field)
    if {norm(r['instruction']) for r in sets['fit']} & {norm(r['instruction']) for r in sets['development']}:
        raise ValueError('Exact instruction overlap')
    manifest={}
    for split,part in sets.items():
        content=b''.join(output_bytes(r) for r in part)
        manifest[split]=dict(count=len(part),sha256=digest(content),by_operation=dict(Counter(r.get('operation','legacy') for r in part)),
                            by_supervision=dict(Counter(r['supervision'] for r in part)))
    lock=dict(salt=SALT,parent_train_sha256=digest(raw),heldout_test_sha256=seal['manifests']['test.jsonl'],
              rule='Development uses new implicit image sources and unique normalized instructions, stratified by construction cohort and action. The two partitions jointly cover the entire outer training split.',
              geometry='Labels are used for training and development scoring only. Boxes use unit image coordinates and movements use fractions of width or height.',
              splits=manifest)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.prepare.lock').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX)
        contents={a.output/(split+'.jsonl'):b''.join(output_bytes(r) for r in part) for split,part in sets.items()}
        contents[a.output/'split-lock.json']=output_bytes(lock)
        for path,content in contents.items():
            if path.exists() and path.read_bytes()!=content:
                raise ValueError('Conflicting training split at '+str(path))
        for path,content in contents.items():
            write_stable(path,content)
    print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':
    main()
