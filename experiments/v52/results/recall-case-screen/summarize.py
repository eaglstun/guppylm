"""Paired family-cluster summaries and immutable raw-output archive."""
import collections,gzip,json,shutil
from pathlib import Path
import numpy as np
from tools.aquarium.v52.compare import paired_difference
from tools.aquarium.v52.data import file_hash
ROOT=Path('/private/tmp/guppylm-dev-apply');RUN=ROOT/'runs/v52/recall-case-screen';DEST=Path(__file__).parent

def counts(rows):
 return {'hits':sum(r['unambiguous_literal_hit'] for r in rows),'n':len(rows),'raw_hits':sum(r['literal_hit'] for r in rows),'alternate_mentions':sum(r['contradictory_alternative'] for r in rows)}

def difference(left,right,case):
 def index(rows):return {(r['family'],r['value_index'],r['mode'],r['case'],r['sample']):r for r in rows if r['case']==case}
 a,b=index(left),index(right);assert a.keys()==b.keys()
 grouped=collections.defaultdict(list)
 for k in a:grouped[k[0]].append(int(b[k]['unambiguous_literal_hit'])-int(a[k]['unambiguous_literal_hit']))
 vals=np.array([np.mean(v) for _,v in sorted(grouped.items())]);rng=np.random.default_rng(89300)
 boot=vals[rng.integers(len(vals),size=(10000,len(vals)))].mean(axis=1)
 return {'difference':float(vals.mean()),'family_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'families':len(vals)}

def main():
 payloads={k:json.loads((RUN/f'{k}-validation.json').read_text()) for k in ('starting','recall','paired')}
 assert len({p['validation_sha256'] for p in payloads.values()})==1
 assert len({json.dumps(p['decoding'],sort_keys=True) for p in payloads.values()})==1
 result={'validation':{},'deltas':{},'development':{},'exposure':{},'checkpoints':{k:p['checkpoint_sha256'] for k,p in payloads.items()}}
 for arm,p in payloads.items():
  result['validation'][arm]={case:{mode:counts([r for r in p['results'] if r['case']==case and (mode=='all' or r['mode']==mode)]) for mode in ('all','direct','separate')} for case in ('normal','uppercase')}
 for a,b in [('starting','recall'),('starting','paired'),('recall','paired')]:
  result['deltas'][f'{b}_minus_{a}']={c:difference(payloads[a]['results'],payloads[b]['results'],c) for c in ('normal','uppercase')}
 starting=json.loads((ROOT/'runs/v52/checks-responsive-v2/development.json').read_text())
 result['development']['starting']=starting['summary']
 for arm in ('recall','paired'):
  report=json.loads((RUN/f'{arm}-development.json').read_text())
  result['development'][arm]={'summary':report['summary'],'delta_from_start':paired_difference(starting,report)}
  train=json.loads((RUN/arm/'model/log.json').read_text());sampling=train['sampling'];source=sampling['recall_screen']
  result['exposure'][arm]={'sampling':sampling,'new_source_target_share':source['target_tokens']/sum(s['target_tokens'] for s in sampling.values()),'last_development_loss':train['history'][-1]['development']['loss']}
  for name in ('recipe.json','log.json'):shutil.copyfile(RUN/arm/'model'/name,DEST/f'{arm}-training-{name}')
  (DEST/f'{arm}-additions.json.gz').write_bytes(gzip.compress((RUN/arm/'additions.json').read_bytes(),mtime=0))
 for path in list(RUN.glob('*-validation.json'))+list(RUN.glob('*-development.json'))+[RUN/'validation.json']:
  (DEST/(path.name+'.gz')).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
 shutil.copyfile(RUN/'audit.json',DEST/'audit.json')
 result['source_files']={p.name:file_hash(p) for p in DEST.glob('*.py')}
 (DEST/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result['validation'],indent=2));print(json.dumps(result['deltas'],indent=2))
if __name__=='__main__':main()
