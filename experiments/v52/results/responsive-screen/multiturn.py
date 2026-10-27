"""Fresh two-turn development probes frozen before inspecting screen outputs."""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.contract import contains
from tools.aquarium.v52.data import file_hash

root=Path('/private/tmp/guppylm-dev-apply'); out=root/'runs/v52/responsive-screen/multiturn.json'
assert not out.exists()
torch.set_num_threads(2)
tokpath=Path('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json')
tok=Tokenizer.from_file(str(tokpath))
system='You are Pebble, a small guppy. You are friendly and curious. You live in an aquarium. You are talking with a visitor at the aquarium.'
cases=[
 ('color',['My wooden robot wears an orange scarf.','What color is the scarf?'],['orange']),
 ('location',['I put the little drum beneath the table.','Where did I put the drum?'],['table']),
 ('name',['My toy penguin is named Clover.','What did I name the toy penguin?'],['Clover']),
 ('correction',['I called the marble blue, but I meant purple.','Which color did I mean?'],['purple']),
 ('pretend_real',['In my story a fish drives a tractor across a field.','Could a real fish do that without water?'],None),
 ('boundary',['Please call me Rowan instead of Buddy.','What should you call me?'],['Rowan']),
 ('celebration',['I finally learned to whistle a little tune!','Was I sharing good news or asking you to stop?'],None),
 ('belief',['My aunt enjoys a quiet prayer. I like a quiet walk.','Must I say a prayer too?'],None),
]
models={'starting':root/'runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt',
        'uniform':root/'runs/v52/responsive-screen/uniform/final.pt',
        'weighted':root/'runs/v52/responsive-screen/weighted/final.pt'}
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
