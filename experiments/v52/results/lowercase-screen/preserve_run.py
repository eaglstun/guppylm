"""Append one run to the training library without replacing existing snapshots."""
import argparse,collections,gzip,json
from pathlib import Path
from tools.aquarium.v52.data import read_manifest,digest,file_hash
SPECIAL={'talkie_vocabulary','uppercase','responsive','grounded_copy','recall_screen'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--library',type=Path,default=Path('experiments/v52/training-library'));a=p.parse_args()
 rows,h,_=read_manifest(a.manifest,'train');recipe=json.loads(a.recipe.read_text());assert recipe['recipe']['train_hash']==h
 index=json.loads((a.library/'index.json').read_text())
 for old in index['datasets'].values():assert file_hash(a.library/old['file'])==old['file_sha256']
 selected={};buckets=collections.defaultdict(list)
 for row in rows:
  if row['source'] in SPECIAL or (row['source']=='talkie' and row.get('augmentation')=='base'):selected[row['uid']]=row
  else:buckets[(row['source'],row.get('augmentation','none'),row.get('persona','unknown'))].append(row)
 for group in buckets.values():
  for row in sorted(group,key=lambda r:digest(['preserve-v1',r['uid']]))[:2]:selected[row['uid']]=row
 kept=sorted(selected.values(),key=lambda r:r['uid']);name=h[:12]+'.jsonl.gz';path=a.library/name
 data=gzip.compress(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept).encode(),mtime=0)
 if path.exists():assert path.read_bytes()==data,'Existing snapshot differs'
 else:path.write_bytes(data)
 info={'path':str(a.manifest),'file':name,'file_sha256':file_hash(path),'original_rows':len(rows),'preserved_rows':len(kept),'source_counts':dict(collections.Counter(r['source'] for r in kept))}
 if h in index['datasets']:assert index['datasets'][h]==info
 else:index['datasets'][h]=info
 run={'run':str(a.recipe.parent),'train_hash':h,'recipe':recipe}
 same=[r for r in index['runs'] if r['run']==run['run']]
 if same:assert same==[run]
 else:index['runs'].append(run)
 (a.library/'index.json').write_text(json.dumps(index,indent=2)+'\n')
 print('Preserved',len(kept),'rows;',len(index['runs']),'runs in library')
if __name__=='__main__':main()
