"""Paired casing/context diagnostics, frozen before inspecting their outputs."""
import collections
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tools.aquarium.v52 import evaluate as ev
from tools.aquarium.v52.data import file_hash, read_manifest

ROOT = Path('/private/tmp/guppylm-dev-apply')
OUT = ROOT/'runs/v52/recall-casing'
SYSTEM = 'You are Pebble, a small guppy. You are friendly and curious. You live in an aquarium. You are talking with a visitor at the aquarium.'
CASES = [
 ('color', 'The scarf on my toy bear is {}.', 'What color is the scarf?', ['orange','purple']),
 ('color', 'I painted my wooden box {}.', 'What color is my wooden box?', ['yellow','green']),
 ('name', 'My toy duck is named {}.', 'What is my toy duck named?', ['Clover','Willa']),
 ('name', 'Please call me {}.', 'What should you call me?', ['Rowan','Lark']),
 ('location', 'I placed the drum {} the table.', 'Where is the drum?', ['under','behind']),
 ('number', 'I have {} shiny pebbles.', 'How many shiny pebbles do I have?', ['three','five']),
 ('mapping', "In this game, 'zembalo' means {}.", 'What does zembalo mean in this game?', ['an umbrella','a lantern']),
 ('mapping', "In this story, 'nuvetti' means {}.", 'What does nuvetti mean in this story?', ['patience','kindness']),
]

@torch.no_grad()
def answer_nll(model,tok,turns,answer):
    prefix,_=ev.aq.chat_ids(tok,SYSTEM,turns,open_assistant=True)
    target=tok.encode(answer).ids+[ev.aq.IM_END]
    ids=torch.tensor([prefix+target])
    logits,_=model.forward_cached(ids[:,:-1],None)
    scores=torch.nn.functional.cross_entropy(logits[0,len(prefix)-1:,:],ids[0,len(prefix):],reduction='mean')
    return float(scores)

@torch.no_grad()
def greedy(model,tok,turns):
    ids,_=ev.aq.chat_ids(tok,SYSTEM,turns,open_assistant=True)
    idx=torch.tensor([ids]); past=None; output=[]
    for _ in range(80):
        logits,past=model.forward_cached(idx if past is None else idx[:,-1:],past)
        nxt=logits[:,-1,:].argmax(-1).reshape(1,1)
        if nxt.item()==ev.aq.IM_END: break
        output.append(nxt.item());idx=torch.cat([idx,nxt],dim=1)
    return tok.decode(output)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    tokpath=Path('/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq/tokenizer.json')
    tok=Tokenizer.from_file(str(tokpath))
    payload={'description':__doc__,'system':SYSTEM,'cases':CASES,'tokenizer_sha256':file_hash(tokpath),
             'script_sha256':file_hash(Path(__file__)),'results':[],'models':{},'tokenization':[]}
    for kind,template,q,values in CASES:
        text=template.format(values[0])
        for case,transform in [('normal',str),('upper',str.upper),('lower',str.lower)]:
            encoded=tok.encode(transform(text));payload['tokenization'].append({'category':kind,'case':case,'text':transform(text),'tokens':encoded.tokens,'ids':encoded.ids,'roundtrip':tok.decode(encoded.ids)})
    for label,path in [('previous',ROOT/'runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt'),('revised',ROOT/'runs/v52/full-responsive-v2-A/best.pt')]:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model=ev.aq.GuppyLM(ev.aq.GuppyConfig(**ck['config']));model.load_state_dict(ck['model']);model.eval()
        payload['models'][label]={'path':str(path),'sha256':file_hash(path),'step':ck['step']}
        for i,(kind,template,q,values) in enumerate(CASES):
            for value in values:
                for case,transform in [('normal',str),('upper',str.upper),('lower',str.lower)]:
                    fact=transform(template.format(value))
                    first=[{'role':'user','content':fact}]
                    self_answer=greedy(model,tok,first)
                    histories={
                      'direct':[{'role':'user','content':fact+' '+q}],
                      'neutral':first+[{'role':'assistant','content':'Okay.'},{'role':'user','content':q}],
                      'self':first+[{'role':'assistant','content':self_answer},{'role':'user','content':q}],
                      'absent':[{'role':'user','content':q}],
                    }
                    for mode,turns in histories.items():
                        scores={v:answer_nll(model,tok,turns,v) for v in values}
                        answer=greedy(model,tok,turns) if mode!='absent' else None
                        ids,_=ev.aq.chat_ids(tok,SYSTEM,turns,open_assistant=True)
                        payload['results'].append({'model':label,'case_id':i,'category':kind,'value':value,'case':case,'mode':mode,
                          'turns':turns,'answer':answer,'hit':ev.contains(answer,value) if answer is not None else None,
                          'answer_nll':scores,'preferred':min(scores,key=scores.get),'prompt_tokens':len(ids)})
            print(label,i,flush=True)
        (OUT/'probes.json').write_text(json.dumps(payload,indent=2)+'\n')
    audit={}
    for label,manifest in [('previous','contract-v3-talkie-uppercase-v2-vocabulary-v1'),('revised','contract-responsive-v2-reviewed')]:
        rows,_,_=read_manifest(ROOT/f'runs/v52/{manifest}/train_A.json','train')
        counts=collections.Counter(); bysource=collections.Counter()
        for row in rows:
            counts['rows']+=1
            users=[t['content'] for t in row['turns'] if t['role']=='user']
            upper=any(t.isupper() for t in users)
            if upper: counts['rows_with_uppercase_user']+=1;bysource[row['source']]+=1
            if len(users)>1: counts['multi_turn_rows']+=1
        audit[label]={'counts':dict(counts),'uppercase_sources':dict(bysource)}
    (OUT/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n')

if __name__=='__main__':main()
