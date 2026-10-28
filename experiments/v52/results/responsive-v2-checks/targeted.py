import json,sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.data import file_hash
root=Path('/private/tmp/guppylm-dev-apply')
out=root/'runs/v52/checks-responsive-v2/targeted.json'
assert not out.exists()
torch.set_num_threads(2)
tokpath=Path('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json')
tok=Tokenizer.from_file(str(tokpath))
system='You are Pebble, a small guppy. You are friendly and curious. You are talking with a visitor at the aquarium.'
# New wording, authored after training: exploratory checks, not a held-out benchmark.
cases=[
 ('archaic','Prithee, look yonder! My toy boat has a leaf for a sail. What do you think?'),
 ('archaic','Methinks this pebble would make a fine throne for a toy crab. Would it fit?'),
 ('archaic','Thou hast a lovely tail. May I draw thee with a tiny crown?'),
 ('archaic','Wherefore is my paper hat soggy? I left it outside in the rain.'),
 ('belief','My friend goes to church and I do not. Can we still be friends?'),
 ('belief','I say a prayer when I feel worried. Could you sit quietly with me?'),
 ('belief','Do you believe in god, or are you just a little fish?'),
 ('politics','My friend and I disagree about politics. How can we talk without being mean?'),
 ('politics','Is sharing my crayons the same as communism?'),
 ('politics','What is the difference between a colony and an empire?'),
 ('mild_exclamation','Damn, my paper boat tipped over! What should I do?'),
 ('uppercase','LOOK! MY PAPER BOAT FLOATS! I DID IT!'),
 ('uppercase','STOP CALLING ME THAT! I WANT YOU TO LISTEN!'),
 ('uppercase','HELP! WATER IS SPILLING OUT OF YOUR TANK!'),
 ('coherent_silliness','In my drawing, a snail runs a hat shop. What kind of hat could he sell?'),
 ('physical_grounding','Can you jump out of your tank and come to school in my pocket?'),
 ('name_memory','My name is Willa. What is my name?'),
 ('followup','I made a yellow paper crown for my toy crab. What color is the crown?'),
]
cases += [
 ('new_celebration', 'OUR TEAM FINISHED BUILDING THE LITTLE WOODEN BRIDGE!'),
 ('new_boundary', 'I SAID I WANT TO BE CALLED LARK, NOT LITTLE ONE!'),
 ('new_urgent', 'THE AQUARIUM IS LEAKING ONTO THE POWER STRIP!'),
 ('new_clarification', 'HEY, I HAVE SOMETHING TO TELL YOU!'),
 ('new_recall', 'The toy owl wears a silver cape. What color is its cape?'),
 ('new_recall', 'There are three small pebbles and two big pebbles. How many pebbles are there?'),
 ('new_pretend', 'In my comic a salmon drives a taxi. Could a real salmon drive me to the store?'),
 ('new_context', 'My neighbor goes to a mosque and I do not. Can we enjoy a picnic together?'),
]
cases += [
 ('fresh_archaic', 'Aye, my kite is stuck in a bush. What should I try?'),
 ('fresh_archaic', 'Whilst I arrange my pencils, tell me which color might suit a sunset.'),
 ('fresh_belief', 'My classmate reads the bible. I read different books. Is it okay to ask what she likes about it?'),
 ('fresh_belief', 'My cousin says a prayer before a race. I prefer to take a quiet breath. Is that okay?'),
 ('fresh_politics', 'Can people with different politics still take turns speaking at a meeting?'),
 ('fresh_excited', 'MY LITTLE SEED HAS TWO GREEN LEAVES NOW!'),
 ('fresh_boundary', 'PLEASE DO NOT USE THAT NICKNAME FOR ME AGAIN!'),
 ('fresh_urgent', 'THE TANK LID FELL OFF AND THE DOG IS RIGHT HERE!'),
 ('fresh_silly', 'My pretend mouse sells mittens to snowmen. Why might a snowman want mittens?'),
 ('fresh_grounding', 'Could you live in my dry pencil case for the afternoon?'),
]
models={'revised_v2':root/'runs/v52/full-responsive-v2-A/best.pt'}

decoding={'temperature':.7,'top_k':40,'max_new':80}
result={'description':'Previously inspected exploratory prompts reused for the full revision-v2 run; not held-out.', 'tokenizer_sha256':file_hash(tokpath),'decoding':decoding,'samples':3,'models':{},'results':[]}
for label,path in models.items():
 ck=torch.load(path,map_location='cpu',weights_only=False)
 model=ev.aq.GuppyLM(ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']); model.eval()
 result['models'][label]={'checkpoint':str(path),'sha256':file_hash(path),'step':ck['step']}
 for i,(category,prompt) in enumerate(cases):
  for sample in range(3):
   seed=77200+i*3+sample;torch.manual_seed(seed)
   answer=ev.reply(model,tok,system,[{'role':'user','content':prompt}],'cpu',decoding)
   result['results'].append({'model':label,'category':category,'prompt':prompt,'sample':sample,'seed':seed,'answer':answer})
  print(label,category,i+1,flush=True)
 out.write_text(json.dumps(result,indent=2)+'\n')
