"""Evaluate frozen novel recall families, identical decoding and fixed neutral history."""
import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.data import read_manifest,file_hash

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('Refusing to overwrite evaluation')
 root=Path('/private/tmp/guppylm-dev-apply');validation=root/'runs/v52/recall-case-screen/validation.json';rows,_,_=read_manifest(validation,'development')
 torch.set_num_threads(2);tokpath=Path('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json');tok=Tokenizer.from_file(str(tokpath))
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=ev.aq.GuppyLM(ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']);model.eval()
 decoding={'temperature':.7,'top_k':40,'max_new':80}
 result={'checkpoint_sha256':file_hash(a.checkpoint),'step':ck['step'],'validation_sha256':file_hash(validation),'tokenizer_sha256':file_hash(tokpath),'decoding':decoding,'samples':2,'context':'Supplied neutral Okay. in separate-turn probes; first reply is not generated.','results':[]}
 for i,row in enumerate(rows):
  info=row['probe'];turns=row['turns'][:-1]
  for sample in range(2):
   # Casing and format variants get the same seed for each family/value/sample.
   seed=89200+info['family']*4+info['value_index']*2+sample;torch.manual_seed(seed)
   answer=ev.reply(model,tok,row['system'],turns,'cpu',decoding)
   hit=ev.contains(answer,info['value']);other=any(ev.contains(answer,v) for v in info['alternatives'] if v!=info['value'])
   result['results'].append({'uid':row['uid'],**info,'sample':sample,'seed':seed,'turns':turns,'answer':answer,'literal_hit':hit,'contradictory_alternative':other,'unambiguous_literal_hit':hit and not other})
  if (i+1)%24==0:print('evaluated',i+1,'/',len(rows),flush=True)
 a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
