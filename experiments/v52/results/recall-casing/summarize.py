"""Summarize fixed diagnostic outputs without rerunning or selecting samples."""
import collections,gzip,json,shutil
from pathlib import Path
root=Path('/private/tmp/guppylm-dev-apply')
source=root/'runs/v52/recall-casing';dest=Path(__file__).parent
x=json.loads((source/'supplement.json').read_text());groups=collections.defaultdict(list)
for row in x['results']:groups[(row['model'],row['case'],row['mode'])].append(row)
summary={'description':'Exploratory paired facts; greedy generation; natural complete-answer conditional NLL, not multiple-choice prompting. Repeated variants are not independent trials.','groups':[],'plumbing':x['plumbing']}
for (model,case,mode),rows in groups.items():
 pairs=collections.defaultdict(list)
 for row in rows:pairs[row['case_id']].append(row)
 summary['groups'].append({'model':model,'case':case,'mode':mode,'n':len(rows),'generated_literal_hits':sum(bool(r['hit']) for r in rows) if mode!='absent' else None,'correct_answer_preference':sum(r['preferred']==r['value'] for r in rows),'both_counterfactual_values_preferred':sum(all(r['preferred']==r['value'] for r in pair) for pair in pairs.values()),'paired_cases':len(pairs)})
original=json.loads((source/'probes.json').read_text())
summary['max_prompt_tokens']=max(r['prompt_tokens'] for r in original['results'])
summary['token_counts']={case:[len(r['ids']) for r in original['tokenization'] if r['case']==case] for case in ('normal','upper','lower')}
summary['models']=original['models']
summary['tokenizer_sha256']=original['tokenizer_sha256']
summary['data_audit']=json.loads((source/'data-audit.json').read_text())
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ('probes.json','supplement.json'):(dest/(name+'.gz')).write_bytes(gzip.compress((source/name).read_bytes(),mtime=0))
shutil.copyfile(source/'data-audit.json',dest/'data-audit.json')
print(json.dumps(summary['token_counts']))
