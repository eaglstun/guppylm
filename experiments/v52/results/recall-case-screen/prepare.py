"""Freeze a matched recall-only versus recall-plus-case continuation screen."""
import json,re,shutil,tempfile
from pathlib import Path
from tokenizers import Tokenizer
from tools.aquarium.v52.data import read_manifest,write_manifest,digest,file_hash,assert_disjoint,normalized
from tools.aquarium.v52.append_context import FROZEN
from tools.aquarium.v52.uppercase import encode_counts
ROOT=Path('/private/tmp/guppylm-dev-apply')
OUT=ROOT/'runs/v52/recall-case-screen'
SYSTEM='You are Pebble, a small guppy. You are friendly and curious. You live in an aquarium. You are talking with a visitor at the aquarium.'
# Each tuple is a context family with two counterfactual values, not independent stories.
TRAIN=[
 ('color','My toy fox has a {} jacket.','What color is the jacket?','The jacket is {}.',['red','blue']),
 ('color','I painted the little cart {}.','What color did I paint the cart?','You painted the cart {}.',['pink','brown']),
 ('color','The crown for my toy frog is {}.','What color is the crown?','The crown is {}. A fine color for a tiny parade.',['gold','white']),
 ('color','My paper kite is {}.','Which color is my paper kite?','Your paper kite is {}.',['black','silver']),
 ('name','My stuffed rabbit is named {}.','What is my stuffed rabbit named?','Your stuffed rabbit is named {}.',['Milo','Nora']),
 ('name','Please use {} as my nickname.','What nickname should you use?','I should call you {}.',['Pip','June']),
 ('name','The captain of my toy ship is {}.','Who is the captain?','The captain is {}.',['Otto','Ada']),
 ('name','I named the pretend dragon {}.','What did I name the dragon?','You named the dragon {}.',['Finn','Iris']),
 ('location','I put my bead {} the cup.','Where is my bead?','Your bead is {} the cup.',['inside','beside']),
 ('location','My sock puppet sits {} the chair.','Where is the sock puppet?','The sock puppet is {} the chair.',['under','behind']),
 ('location','I left my notebook {} the tray.','Where is the notebook?','The notebook is {} the tray.',['on','beneath']),
 ('location','The toy castle is {} the basket.','Where is the toy castle?','The toy castle is {} the basket.',['near','inside']),
 ('number','I made {} clay stars.','How many clay stars did I make?','You made {} clay stars.',['two','four']),
 ('number','There are {} buttons in my jar.','How many buttons are in my jar?','There are {} buttons in your jar.',['six','eight']),
 ('number','I drew {} smiling snails.','How many snails did I draw?','You drew {} smiling snails.',['one','three']),
 ('number','My toy train has {} cars.','How many cars does my toy train have?','Your toy train has {} cars.',['five','seven']),
 ('mapping',"In my game, 'bimwock' means {}.",'What does bimwock mean in my game?',"In your game, bimwock means {}.",['a ceramic mug','a cotton flag']),
 ('mapping',"I use 'pelmish' as my story word for {}.",'What is pelmish in my story?',"In your story, pelmish means {}.",['a wooden spoon','a woolen cap']),
 ('mapping',"For our pretend game, 'zorbette' stands for {}.",'What does zorbette stand for here?',"Here, zorbette stands for {}.",['courage','kindness']),
 ('mapping',"In my silly dictionary, 'frandip' means {}.",'What does frandip mean in that dictionary?',"In that dictionary, frandip means {}.",['a copper bell','a rubber stamp']),
 ('correction','I first said my ribbon was green. That was wrong: it is {}.','What color is my ribbon really?','Your ribbon is {}.',['red','blue']),
 ('correction','I called my toy tiger Max by mistake. Its name is {}.','What is the correct name of my toy tiger?','Your toy tiger is named {}.',['Milo','Nora']),
 ('correction','I thought I had ten shells, but I counted again and have {}.','How many shells do I actually have?','You actually have {} shells.',['two','four']),
 ('correction','I said the coin was on the box. I moved it {} the box.','Where is the coin now?','The coin is now {} the box.',['under','behind']),
]
VALID=[
 ('color','Look at the {} waistcoat on this toy badger.','Tell me the waistcoat color.','The waistcoat is {}.',['orange','purple']),
 ('color','For the miniature windmill, I chose {} paint.','Which paint color did I choose for it?','You chose {} paint.',['yellow','green']),
 ('name','This plush otter answers to {}.','Remind me what the plush otter is called.','The plush otter is called {}.',['Tessa','Hugo']),
 ('name','The name on my pretend astronaut badge reads {}.','Read back the name from my badge.','Your badge says {}.',['Vera','Leon']),
 ('location','The bookmark ended up {} the cushion.','Can you repeat where the bookmark ended up?','The bookmark is {} the cushion.',['under','behind']),
 ('location','For safekeeping, my thimble went {} the drawer.','Describe where I put my thimble.','Your thimble is {} the drawer.',['inside','beside']),
 ('number','My latest drawing shows exactly {} jellybeans.','Repeat the number of jellybeans I drew.','You drew {} jellybeans.',['four','six']),
 ('number','At the finish, I had collected {} pinecones.','How many pinecones did I finish with?','You finished with {} pinecones.',['three','five']),
 ('mapping',"For today's tale, I have decided that 'vemlora' will refer to {}.",'Tell me what vemlora refers to in this tale.',"In this tale, vemlora refers to {}.",['a velvet mitten','a silver buckle']),
 ('mapping',"Let 'tobrelki' be our invented term for {} in this puzzle.",'Repeat the meaning assigned to tobrelki.',"In this puzzle, tobrelki means {}.",['patience','curiosity']),
 ('correction','Forget what I said about a black sled; the sled is actually {}.','Give me the corrected sled color.','The sled is actually {}.',['orange','purple']),
 ('correction','The toy walrus used to be called Toby, but I renamed it {}.','Tell me the new name, rather than the old one.','Its new name is {}.',['Tessa','Hugo']),
]

def examples(specs,split,cases):
 result=[]
 for i,(kind,fact,question,answer,values) in enumerate(specs):
  for j,value in enumerate(values):
   for mode in ('direct','separate'):
    turns=([{'role':'user','content':fact.format(value)+' '+question},{'role':'assistant','content':answer.format(value)}] if mode=='direct' else
      [{'role':'user','content':fact.format(value)},{'role':'assistant','content':'Okay.'},{'role':'user','content':question},{'role':'assistant','content':answer.format(value)}])
    for case in cases:
     ts=[dict(t) for t in turns]
     if case=='uppercase':ts[0]['content']=ts[0]['content'].upper()
     key=[split,i,j,mode,case];parent=digest([split,i,j]);uid=digest(['recall-case-screen-v1',key,ts])
     result.append({'uid':uid,'parent_id':parent,'group_id':digest([split,i]),'split':split,'source':'recall_screen',
       'augmentation':case,'name':'Pebble','persona':'guppy','system':SYSTEM,'turns':ts,'exchanges':len(ts)//2,
       'probe':{'family':i,'value_index':j,'category':kind,'value':value,'alternatives':values,'mode':mode,'case':case},
       'provenance':{'version':'recall-case-screen-v1','construction':'controlled paired template'}})
 return result

def main():
 if OUT.exists():raise ValueError('Refusing existing experiment output')
 base=ROOT/'runs/v52/contract-responsive-v2-reviewed'
 rows,h,meta=read_manifest(base/'train_A.json','train')
 evaluation=[r for name in FROZEN[1:] for r in read_manifest(base/name)[0]]
 contract=json.loads((base/'contract.json').read_text());tokpath=Path('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json');tok=Tokenizer.from_file(str(tokpath))
 forbidden=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(normalized(x)) for x in sum(contract['reserved'].values(),[])+['axolotl','cuttlefish'])+r')(?!\w)')
 arms={'recall':examples(TRAIN,'train',('normal',)),'paired':examples(TRAIN,'train',('normal','uppercase'))}
 validation=examples(VALID,'development',('normal','uppercase'))
 known={normalized(t['content']) for r in rows+evaluation for t in r['turns'] if t['role']=='user'}
 probes={normalized(q) for c in contract['cases'] for q in c['users']}
 probes|={normalized(t['content']) for c in contract['cases'] for t in c['context'] if t['role']=='user'}
 for r in validation:
  assert all(normalized(t['content']) not in known for t in r['turns'] if t['role']=='user'),'Validation question already present'
 token_audit={}
 for label,additions in {**arms,'validation':validation}.items():
  assert_disjoint(additions,rows+evaluation)
  totals={'rows':len(additions),'input_tokens':0,'target_tokens':0}
  for r in additions:
   assert not forbidden.search(normalized(r['system']+' '+str(r['turns'])))
   assert not any(normalized(t['content']) in probes for t in r['turns'] if t['role']=='user')
   n,target,truncated=encode_counts(r,tok,512);assert target and not truncated
   totals['input_tokens']+=n;totals['target_tokens']+=target
  token_audit[label]=totals
 for additions in arms.values():assert_disjoint(additions,validation)
 # Paired rows differ only in first-user casing and lineage metadata.
 for a,b,c in zip(arms['recall'],arms['paired'][::2],arms['paired'][1::2]):
  assert a==b and a['turns'][1:]==c['turns'][1:]
  assert a['turns'][0]['content'].upper()==c['turns'][0]['content']
 with tempfile.TemporaryDirectory(prefix='recall-case-',dir=OUT.parent) as temp:
  stage=Path(temp)
  for label,additions in arms.items():
   dest=stage/label
   write_manifest(dest/'train_A.json',rows+additions,{**meta,'recall_case_screen':{'base_manifest_sha256':h,'builder_sha256':file_hash(Path(__file__)),'arm':label}})
   for name in FROZEN:shutil.copyfile(base/name,dest/name)
   (dest/'additions.json').write_text(json.dumps(additions,indent=2)+'\n')
  write_manifest(stage/'validation.json',validation,{'builder_sha256':file_hash(Path(__file__)),'purpose':'new exploratory validation held out from add-on training'})
  audit={'base_hash':h,'tokenizer_hash':file_hash(tokpath),'tokens':token_audit,'frozen_files':{n:file_hash(base/n) for n in FROZEN},'notes':'Validation nouns, names, meanings and wording are distinct from add-on templates; common words may occur in inherited training/pretraining.'}
  (stage/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');stage.rename(OUT)
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
