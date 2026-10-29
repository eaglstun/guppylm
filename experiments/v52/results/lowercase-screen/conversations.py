"""Fish conversations before/after lowercase continuation with matched normalized context."""
import importlib.util,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.data import file_hash
ROOT=Path('/private/tmp/guppylm-dev-apply');OUT=ROOT/'runs/v52/lowercase-screen/conversations.json'
spec=importlib.util.spec_from_file_location('scenes',ROOT/'experiments/v52/results/fish-conversation-review/probe.py');scenes=importlib.util.module_from_spec(spec);spec.loader.exec_module(scenes)
def main():
 if OUT.exists():raise ValueError('Refusing existing conversation outputs')
 torch.set_num_threads(2);tokpath=ROOT/'runs/v52/contract-lowercase-v1-reviewed/data/tokenizer.json';tok=Tokenizer.from_file(str(tokpath));decoding={'temperature':.7,'top_k':40,'max_new':80}
 result={'text_case':'lowercase','description':__doc__,'decoding':decoding,'tokenizer_sha256':file_hash(tokpath),'script_sha256':file_hash(Path(__file__)),'models':{},'conversations':[]}
 for label,path in [('starting',ROOT/'runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt'),('lowercase_600',ROOT/'runs/v52/lowercase-screen/model/final.pt')]:
  ck=torch.load(path,map_location='cpu',weights_only=False);model=ev.aq.GuppyLM(ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']);model.eval();result['models'][label]={'path':str(path),'sha256':file_hash(path),'step':ck['step']}
  for i,(scene,users) in enumerate(scenes.SCENES):
   turns=[]
   for j,user in enumerate(users):
    turns.append({'role':'user','content':user});torch.manual_seed(90400+i*3+j)
    answer=ev.reply(model,tok,scenes.SYSTEM,turns,'cpu',decoding,'lowercase');turns.append({'role':'assistant','content':answer})
   result['conversations'].append({'model':label,'scene':scene,'turns':turns})
  OUT.write_text(json.dumps(result,indent=2)+'\n');print(label,'done',flush=True)
if __name__=='__main__':main()
