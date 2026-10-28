"""Follow-up: sentence-case normalization, natural answer scoring, framing/cache checks."""
import importlib.util,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
spec=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('probe.py'));p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
torch.set_num_threads(2)
tok=Tokenizer.from_file('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json')
file=p.OUT/'supplement.json';assert not file.exists()
old=json.loads((p.OUT/'probes.json').read_text())
answers=['The scarf is {}.','Your wooden box is {}.','Your toy duck is named {}.','I should call you {}.','The drum is {} the table.','You have {} shiny pebbles.',"In this game, zembalo means {}.","In this story, nuvetti means {}."]
result={'description':__doc__,'script_sha256':p.file_hash(Path(__file__)),'results':[],'plumbing':[]}
for label,info in old['models'].items():
 ck=torch.load(info['path'],map_location='cpu',weights_only=False);model=p.ev.aq.GuppyLM(p.ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']);model.eval()
 for row in old['results']:
  if row['model']!=label:continue
  vals=p.CASES[row['case_id']][3]
  scores={v:p.answer_nll(model,tok,row['turns'],answers[row['case_id']].format(v)) for v in vals}
  result['results'].append({**{k:row[k] for k in ('model','case_id','value','case','mode')},'answer_nll':scores,'preferred':min(scores,key=scores.get),'answer':row['answer'],'hit':row['hit']})
 for i,(kind,template,q,values) in enumerate(p.CASES):
  for value in values:
   # Reversible diagnostic on a copy; does not change input processing in the app.
   fact=template.format(value).upper().capitalize()
   first=[{'role':'user','content':fact}]
   self_answer=p.greedy(model,tok,first)
   for mode,turns in [('direct',[{'role':'user','content':fact+' '+q}]),('neutral',first+[{'role':'assistant','content':'Okay.'},{'role':'user','content':q}]),('self',first+[{'role':'assistant','content':self_answer},{'role':'user','content':q}])]:
    answer=p.greedy(model,tok,turns);scores={v:p.answer_nll(model,tok,turns,answers[i].format(v)) for v in values}
    result['results'].append({'model':label,'case_id':i,'value':value,'case':'sentence_normalized','mode':mode,'turns':turns,'answer':answer,'hit':p.ev.contains(answer,value),'answer_nll':scores,'preferred':min(scores,key=scores.get)})
 turns=[{'role':'user','content':'My scarf is orange.'},{'role':'assistant','content':'Your scarf is orange.'},{'role':'user','content':'What color is it?'}]
 ids,_=p.ev.aq.chat_ids(tok,p.SYSTEM,turns,open_assistant=True)
 with torch.no_grad():
  x=torch.tensor([ids]);full,_=model(x);cached,_=model.forward_cached(x,None)
  _,past=model.forward_cached(x[:,:-1],None);last,_=model.forward_cached(x[:,-1:],past)
  d1=float((full-cached).abs().max());d2=float((full[:,-1,:]-last[:,-1,:]).abs().max())
 assert d1<1e-4 and d2<1e-4
 row={'system':p.SYSTEM,'turns':turns[:-1]};x,y=p.ev.aq.encode_chat(tok,row,512)
 target=[t for t in y if t];expected=tok.encode(turns[1]['content']).ids+[p.ev.aq.IM_END]
 assert target==expected
 result['plumbing'].append({'model':label,'full_cached_max_error':d1,'incremental_max_error':d2,'assistant_target_mask_exact':True})
 print(label,'done',flush=True)
file.write_text(json.dumps(result,indent=2)+'\n')
