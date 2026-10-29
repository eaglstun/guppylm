"""Keep stratified samples and distinctive additions from every local v5.2 run."""
import collections,gzip,json
from pathlib import Path
from tools.aquarium.v52.data import read_manifest,digest,file_hash
ROOT=Path('/private/tmp/guppylm-dev-apply');OUT=ROOT/'experiments/v52/training-library'
SPECIAL={'talkie_vocabulary','uppercase','responsive','grounded_copy','recall_screen'}
def main():
 if OUT.exists():raise ValueError('Refusing to replace preserved library')
 recipes=sorted((ROOT/'runs/v52').rglob('recipe.json'))
 required={json.loads(p.read_text())['recipe']['train_hash'] for p in recipes}
 manifests={}
 for path in sorted((ROOT/'runs/v52').rglob('train_A.json')):
  rows,h,meta=read_manifest(path,'train')
  if h in required and h not in manifests:manifests[h]=(path,rows,meta)
 assert required==set(manifests),'Missing run dataset'
 OUT.mkdir()
 index={'selection':'All distinctive add-on sources and original Talkie base conversations; otherwise up to two deterministic rows per source/augmentation/persona bucket. Historical samples are not automatically approved for future training.','datasets':{},'runs':[]}
 for h,(path,rows,meta) in manifests.items():
  buckets=collections.defaultdict(list);keep={}
  for r in rows:
   if r['source'] in SPECIAL or (r['source']=='talkie' and r.get('augmentation')=='base'):keep[r['uid']]=r
   else:buckets[(r['source'],r.get('augmentation','none'),r.get('persona','unknown'))].append(r)
  for key,group in buckets.items():
   for r in sorted(group,key=lambda r:digest(['preserve-v1',r['uid']]))[:2]:keep[r['uid']]=r
  selected=sorted(keep.values(),key=lambda r:r['uid']);name=h[:12]+'.jsonl.gz'
  (OUT/name).write_bytes(gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode(),mtime=0))
  index['datasets'][h]={'path':str(path.relative_to(ROOT)),'file':name,'file_sha256':file_hash(OUT/name),'original_rows':len(rows),'preserved_rows':len(selected),'source_counts':dict(collections.Counter(r['source'] for r in selected))}
 for path in recipes:
  payload=json.loads(path.read_text());index['runs'].append({'run':str(path.parent.relative_to(ROOT)),'train_hash':payload['recipe']['train_hash'],'recipe':payload})
 (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
 print(json.dumps({h:d['preserved_rows'] for h,d in index['datasets'].items()},indent=2));print(len(recipes),'runs preserved')
if __name__=='__main__':main()
