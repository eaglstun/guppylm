"""Fresh exploratory word-definition probes authored after training, before sampling; tiny diagnostic, not a benchmark."""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.contract import contains
from tools.aquarium.v52.data import file_hash

root=Path('/private/tmp/guppylm-dev-apply'); out=root/'runs/v52/checks-responsive-v2/grounding.json'
assert not out.exists()
torch.set_num_threads(2)
tokpath=Path('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json')
tok=Tokenizer.from_file(str(tokpath))
system='You are Pebble, a small guppy. You are friendly and curious. You live in an aquarium. You are talking with a visitor at the aquarium.'
cases=[
 ('explicit_object',["In our game, 'zembalo' means a striped umbrella.","What does zembalo mean in our game?"],['striped umbrella']),
 ('explicit_abstract',["For this story, 'nuvetti' means patience.","What does nuvetti mean here?"],['patience']),
 ('corrected_mapping',["I said 'dorbina' meant a kettle. I was wrong: in this game it means a lantern.","What does dorbina mean now?"],['lantern']),
 ('clarify_first',["I found a fepnora today.","Fepnora is my pretend word for a velvet ribbon. What does it mean?"],['velvet ribbon']),
]

models={'previous':root/'runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt','revised_v2':root/'runs/v52/full-responsive-v2-A/best.pt'}

result={'description':__doc__,'decoding':{'temperature':.7,'top_k':40,'max_new':80},'samples':2,'models':{},'results':[]}
for name,path in models.items():
 ck=torch.load(path,map_location='cpu',weights_only=False)
 model=ev.aq.GuppyLM(ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']);model.eval()
 result['models'][name]={'sha256':file_hash(path),'path':str(path)}
 for i,(kind,users,expected) in enumerate(cases):
  for casing in ('normal','uppercase'):
   for sample in range(2):
    torch.manual_seed(82100+i*2+sample)
    turns=[]
    for j,user in enumerate(users):
     turns.append({'role':'user','content':user.upper() if j==0 and casing=='uppercase' else user})
     answer=ev.reply(model,tok,system,turns,'cpu',result['decoding'])
     turns.append({'role':'assistant','content':answer})
    result['results'].append({'model':name,'category':kind,'case':casing,'sample':sample,'turns':turns,
       'expected':expected,'literal_hit':any(contains(answer,x) for x in expected) if expected else None})
 print(name,'done',flush=True)
 out.write_text(json.dumps(result,indent=2)+'\n')
