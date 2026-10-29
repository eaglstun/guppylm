"""Reference the exact int8-rounded checkpoint for browser export verification."""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.data import file_hash
root=Path('/private/tmp/guppylm-dev-apply');out=root/'runs/v52/lowercase-web/reference.json'
assert not out.exists();torch.set_num_threads(2)
ck=torch.load(root/'runs/v52/lowercase-web/int8.pt',map_location='cpu',weights_only=False)
model=ev.aq.GuppyLM(ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']);model.eval()
tok=Tokenizer.from_file(str(root/'web/model/tokenizer.json'))
system='You are Pebble, a guppy who lives in a small aquarium. You are curious and excitable.'
prompts=['Hi little fish!','LOOK! I MADE A BRIDGE!','Can I sit quietly and watch you swim?','Imagine a snail wearing a tiny hat.','Hello Élodie 🐟 — 你好!','My toy crab is named Clover. What is it named?']
result={'checkpoint_sha256':file_hash(root/'runs/v52/lowercase-screen/model/final.pt'),'binary_sha256':file_hash(root/'web/model/guppy.bin'),'cases':[]}
with torch.no_grad():
 for user in prompts:
  turns=[{'role':'user','content':user}];normalized=[{'role':'user','content':user.lower()}]
  ids,_=ev.aq.chat_ids(tok,system.lower(),normalized,open_assistant=True);x=torch.tensor([ids]);logits,past=model.forward_cached(x,None)
  first=logits[0,-1].tolist();tokens=[]
  for _ in range(24):
   nxt=int(logits[0,-1].argmax())
   if nxt==ev.aq.IM_END:break
   tokens.append(nxt);logits,past=model.forward_cached(torch.tensor([[nxt]]),past)
  result['cases'].append({'system':system,'turns':turns,'ids':ids,'first_logits':first,'greedy_tokens':tokens})
out.write_text(json.dumps(result)+'\n');print('Prepared',len(prompts),'reference prompts')
