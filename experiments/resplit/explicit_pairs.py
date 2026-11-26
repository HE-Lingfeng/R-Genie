"""Register auxiliary plan supervision, excluding every validation and test source/target."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifests',type=Path,default=Path('artifacts/reditbench-resplit-v2'))
    args=parser.parse_args()
    split,data=args.manifests,args.data
    records={k:list(map(json.loads,(split/(k+'.jsonl')).read_text().splitlines())) for k in ('train','validation','test')}
    seen={r['id'] for r in records['train']}
    hold=records['validation']+records['test']
    blocked={r['id'] for r in hold}
    held_pixels={r[k] for r in hold for k in ('source_sha256','target_sha256')}
    pairs=[]
    for i,v in json.loads((data/'training_refering_sentences.json').read_text()).items():
        if i in blocked:continue
        fn='COCO_train2014_'+i.zfill(12)
        src='image/'+fn+'.jpg';tar='target/'+fn+'_edit.jpg'
        if not ((data/src).is_file() and (data/tar).is_file()):continue
        s,t=digest(data/src),digest(data/tar)
        if s in held_pixels or t in held_pixels:raise ValueError('Image content overlap '+i)
        assert v['instruction'].strip()
        pairs.append(dict(id=i,source=src,target=tar,source_sha256=s,target_sha256=t,plan=v['instruction'].strip(),previous_implicit_train=i in seen))
    pairs.sort(key=lambda r:int(r['id']))
    assert len(pairs)==343 and sum(r['previous_implicit_train'] for r in pairs)==161
    path=split/'aux-explicit-train.jsonl'
    content=''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in pairs)
    if path.exists() and path.read_text()!=content:raise ValueError('Auxiliary set differs')
    path.write_text(content)
    pre=dict(experiment='explicit-plan decoder continuation',input_manifest=digest(path),unique_explicit=343,
        additional_unique_train=182,total_unique_optimization=749,
        holdout_sha256=digest(split/'test.jsonl'),selection_data='100-image validation only',
        updates='3 epochs on 343 explicit pairs, generator LoRA only, lr 2e-5, no transformer-block C; spatial ensemble stays frozen',
        select_checkpoint='lowest validation weak-region-weighted flow loss with fixed noise; initial checkpoint eligible',
        test_selection=False)
    (split/'aux-protocol.json').write_text(json.dumps(pre,indent=2)+'\n')
    print(len(pairs),pre['input_manifest'])


if __name__=='__main__':
    main()
