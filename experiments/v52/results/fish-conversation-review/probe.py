"""Small exploratory fish conversations, with original/lowercase user-input comparison."""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.data import file_hash
ROOT=Path('/private/tmp/guppylm-dev-apply');OUT=ROOT/'runs/v52/fish-conversation-review'
SYSTEM='You are Pebble, a small guppy. You are friendly and curious. You live in an aquarium. You are talking with a visitor at the aquarium.'
SCENES=[
 ('hello',['Hi little fish! How is life behind the glass?','Your tank looks cozy. Which bit do you like?','I will sit here and watch you swim for a minute.']),
 ('pretend_shop',['Imagine a snail opening a tiny hat shop.','His first customer is a crab with a very large head.','What should we name the shop?']),
 ('show_and_tell',['I made a paper crown for my toy frog!','It keeps slipping over his eyes. He looks very serious.','Maybe he needs a smaller crown.']),
 ('curiosity',['What do you think rain sounds like from inside your tank?','I think it sounds like tiny feet on a roof.','Can you make up a little rain sound?']),
 ('quiet_company',['I had a long day. Can I just hang out here?','Thanks. I do not really feel like talking much.','Your little fins are nice to watch.']),
 ('fish_story',['Let us tell a silly story about a fish librarian.','One morning all the books start blowing bubbles.','How does our fish librarian sort that out?']),
 ('excited',['LOOK! I MADE A LITTLE BRIDGE OUT OF STICKS!','A TOY MOUSE CAN WALK ACROSS IT!','WHAT SHOULD WE PUT ON THE OTHER SIDE?']),
 ('respectful',['My friend likes to say a prayer. I like to sit quietly.','We both like visiting the aquarium.','Maybe we can just enjoy watching you together.']),
]
MODELS={'previous':ROOT/'runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt','revised':ROOT/'runs/v52/full-responsive-v2-A/best.pt','recall':ROOT/'runs/v52/recall-case-screen/recall/model/final.pt','paired':ROOT/'runs/v52/recall-case-screen/paired/model/final.pt'}
def main():
 OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 tokpath=Path('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json');tok=Tokenizer.from_file(str(tokpath));decoding={'temperature':.7,'top_k':40,'max_new':80}
 result={'description':__doc__,'system':SYSTEM,'scenes':SCENES,'decoding':decoding,'tokenizer_sha256':file_hash(tokpath),'script_sha256':file_hash(Path(__file__)),'models':{},'conversations':[],'limitations':'One sample per scene/condition; subjective exploratory review. Only user turns normalized; existing checkpoint and system prompt unchanged. No lowercase-training claim.'}
 for model_name,path in MODELS.items():
  ck=torch.load(path,map_location='cpu',weights_only=False);model=ev.aq.GuppyLM(ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']);model.eval();result['models'][model_name]={'checkpoint':str(path),'sha256':file_hash(path)}
  for transform in ('original','lowercase'):
   for i,(scene,users) in enumerate(SCENES):
    turns=[];token_counts=[]
    for j,user in enumerate(users):
     text=user.lower() if transform=='lowercase' else user;turns.append({'role':'user','content':text});token_counts.append(len(tok.encode(text).ids));torch.manual_seed(90400+i*3+j)
     answer=ev.reply(model,tok,SYSTEM,turns,'cpu',decoding);turns.append({'role':'assistant','content':answer})
    result['conversations'].append({'model':model_name,'transform':transform,'scene':scene,'original_users':users,'turns':turns,'user_token_counts':token_counts})
   print(model_name,transform,'done',flush=True)
  (OUT/'conversations.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
