"""Versioned lowercase dialogue candidate and bounded lowercase story replay pool."""
import argparse,collections,json,shutil,tempfile
from pathlib import Path
import numpy as np
from .data import read_manifest,write_manifest,digest,file_hash,assert_disjoint
from .append_context import FROZEN

def lower_chat(system, turns):
    return system.lower(), [{**t, 'content': t['content'].lower()} for t in turns]

def lower_row(row):
    system, turns = lower_chat(row['system'], row['turns'])
    return {**row, 'system':system, 'turns':turns,
            'uid':digest([row['uid'],'lowercase-v1',system,turns]),
            'case_transform':{'from_uid':row['uid'],'rule':'lowercase system and conversation text'}}

def lower_story_tokens(tokens, tok):
    # Complete documents, not random slices through a UTF-8 character or BPE word.
    text=tok.decode(list(tokens),skip_special_tokens=False).lower()
    ids=tok.encode(text,add_special_tokens=False).ids
    if not ids or ids[-1]!=3 or ids.count(3)!=1:
        raise ValueError('Expected one complete story with end marker')
    if tok.decode(ids,skip_special_tokens=False)!=text:
        raise ValueError('Story text failed tokenizer round-trip')
    return ids

def story_pool(source,out,tok,count,seed):
    stream=np.memmap(source,dtype=np.uint16,mode='r')
    ends=np.concatenate([np.flatnonzero(stream[i:i+1_000_000]==3)+i for i in range(0,len(stream),1_000_000)])
    if len(ends)<count:raise ValueError('Insufficient complete stories')
    selected=np.sort(np.random.default_rng(seed).choice(len(ends),count,replace=False));total=0
    with out.open('xb') as handle:
        for j in selected:
            start=0 if j==0 else int(ends[j-1])+1
            ids=lower_story_tokens(stream[start:int(ends[j])+1],tok)
            np.asarray(ids,dtype=np.uint16).tofile(handle);total+=len(ids)
    return {'source_sha256':file_hash(source),'output_sha256':file_hash(out),'seed':seed,
            'documents':count,'document_indices':selected.tolist(),'tokens':total,'available_documents':len(ends)}

def build(base,out,data,carry,tok,train_stories=4096,valid_stories=512):
    if out.exists():raise ValueError('Refusing existing output')
    original,h,meta=read_manifest(base/'train_A.json','train')
    extras=[json.loads(l) for l in carry.read_text().splitlines() if l.strip()]
    evaluation=[r for name in FROZEN[1:] for r in read_manifest(base/name)[0]]
    assert_disjoint(extras,evaluation)
    kept={};duplicates=[];added=[]
    for index,old in enumerate(original+extras):
        # Historical carry-forward content keeps the current habitat wording.
        row=lower_row(old)
        row['system']=row['system'].replace('you only know about life in ','you live in ')
        row['uid']=digest([old['uid'],'lowercase-v1',row['system'],row['turns']])
        key=digest([row['system'],row['turns']])
        if key in kept:
            duplicates.append({'from_uid':old['uid'],'retained_uid':kept[key]['uid']})
        else:
            kept[key]=row
            if index>=len(original):added.append(old['uid'])
    rows=list(kept.values())
    with tempfile.TemporaryDirectory(prefix='lowercase-',dir=out.parent) as tmp:
        stage=Path(tmp);frozen=stage/'original-evaluation';frozen.mkdir();newmeta={**meta,'lowercase_v1':{'base_hash':h,'carry_sha256':file_hash(carry),'builder_sha256':file_hash(Path(__file__))}}
        for name in FROZEN:shutil.copyfile(base/name,frozen/name)
        eval_lower=[]
        for name in FROZEN[1:]:
            rs,_,m=read_manifest(base/name);rs=[lower_row(r) for r in rs];eval_lower+=rs
            write_manifest(stage/name,rs,{**m,'text_case':'lowercase'})
        assert_disjoint(rows,eval_lower)
        manifest_hash=write_manifest(stage/'train_A.json',rows,newmeta)
        # Contract remains frozen; evaluate with explicit --text-case lowercase.
        shutil.copyfile(base/'contract.json',stage/'contract.json')
        lowerdata=stage/'data';lowerdata.mkdir();shutil.copyfile(data/'tokenizer.json',lowerdata/'tokenizer.json')
        stories={}
        for split,count,seed in [('train',train_stories,90501),('valid',valid_stories,90502)]:
            stories[split]=story_pool(data/f'ts_{split}.bin',lowerdata/f'ts_{split}.bin',tok,count,seed)
        audit={'base_rows':len(original),'rows':len(rows),'manifest_sha256':manifest_hash,'carry_forward_added':added,
               'duplicates':duplicates,'sources':dict(collections.Counter(r['source'] for r in rows)),
               'tokenizer_sha256':file_hash(data/'tokenizer.json'),'stories':stories,
               'original_evaluation':{name:file_hash(base/name) for name in FROZEN},
               'note':'Lowercase continuation data only; initialization was pretrained on mixed-case text. Bounded uniformly selected complete-story replay pool, not the full story corpus.'}
        (stage/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');stage.rename(out)
    return {k:v for k,v in audit.items() if k not in ('duplicates','stories')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--carry',type=Path,required=True);a=p.parse_args()
    from tokenizers import Tokenizer
    print(json.dumps(build(a.base,a.out,a.data,a.carry,Tokenizer.from_file(str(a.data/'tokenizer.json'))),indent=2))
if __name__=='__main__':main()
