"""Lowercase lineage, story boundaries, UI immutability and comparison isolation."""
import json,subprocess,tempfile,unittest
import numpy as np
from pathlib import Path
from tools.aquarium.v52.lowercase import lower_chat,lower_row,lower_story_tokens,build
from tools.aquarium.v52.data import write_manifest
from tools.aquarium.v52.compare import paired_difference

class LowercaseTests(unittest.TestCase):
    def test_preserves_original_identity_and_history(self):
        old={'uid':'row','parent_id':'parent','group_id':'group','system':'You are PEBBLE.',
             'turns':[{'role':'user','content':'HI, I’m ÉLODIE! 🐟'},{'role':'assistant','content':'Hello!'}]}
        new=lower_row(old)
        self.assertEqual(old['turns'][0]['content'],'HI, I’m ÉLODIE! 🐟')
        self.assertEqual(new['turns'][0]['content'],'hi, i’m élodie! 🐟')
        self.assertEqual(new['parent_id'],'parent');self.assertEqual(new['group_id'],'group')
        self.assertNotEqual(new['uid'],old['uid'])
        self.assertEqual(new['case_transform']['from_uid'],'row')
        self.assertEqual(lower_chat(new['system'],new['turns']),(new['system'],new['turns']))

    def test_complete_story_boundary_and_roundtrip(self):
        class Tok:
            def decode(self,ids,skip_special_tokens=False):
                return ''.join({3:'<|endoftext|>',4:'HI',5:'hi'}[i] for i in ids)
            def encode(self,text,add_special_tokens=False):
                return type('Encoding',(),{'ids':[5,3] if text.endswith('<|endoftext|>') else [5]})()
        self.assertEqual(lower_story_tokens([4,3],Tok()),[5,3])
        with self.assertRaises(ValueError):lower_story_tokens([4],Tok())
        with self.assertRaises(ValueError):lower_story_tokens([4,3,3],Tok())

    def test_carry_forward_dedup_counts_only_new_rows(self):
        class Tok:
            def decode(self,ids,skip_special_tokens=False):
                return ''.join({3:'<|endoftext|>',4:'HI',5:'hi'}[int(i)] for i in ids)
            def encode(self,text,add_special_tokens=False):
                return type('Encoding',(),{'ids':[5,3]})()
        def row(key):
            return {'uid':key,'parent_id':key,'group_id':key,'source':'fixture','split':'train',
                    'system':'You are Pebble.','turns':[{'role':'user','content':key},{'role':'assistant','content':'Hello!'}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';data=root/'data';base.mkdir();data.mkdir()
            write_manifest(base/'train_A.json',[row('original')],{})
            for name in ('development','historical','compression_development'):
                write_manifest(base/(name+'.json'),[{**row(name),'split':name}],{})
            (base/'contract.json').write_text('{}')
            (data/'tokenizer.json').write_text('{}')
            for name in ('train','valid'):np.asarray([4,3,4,3],dtype=np.uint16).tofile(data/('ts_'+name+'.bin'))
            carry=root/'carry.jsonl';carry.write_text(json.dumps(row('original'))+'\n'+json.dumps(row('new'))+'\n')
            result=build(base,root/'out',data,carry,Tok(),1,1)
            self.assertEqual(result['carry_forward_added'],['new'])
            self.assertEqual(result['rows'],2)

    def test_case_conditions_cannot_be_compared_as_equal(self):
        a={'contract_sha256':'x','split':'development','decoding':{},'results':[]}
        with self.assertRaisesRegex(ValueError,'case transformations'):
            paired_difference(a,{**a,'text_case':'lowercase'})
        self.assertEqual(paired_difference(a,{**a,'text_case':'original'}),{})

    def test_browser_matches_python_and_does_not_mutate_ui_turns(self):
        root=Path(__file__).resolve().parents[1]
        system='You are PEBBLE.';turns=[{'role':'user','content':'HELLO ÉLODIE 🐟'},{'role':'assistant','content':'HELLO!'}]
        code='''import {readFileSync} from 'node:fs';
const {lowercaseChat}=await import('data:text/javascript;base64,'+readFileSync('./web/guppy.js').toString('base64'));
const input=JSON.parse(process.argv[1]); const before=JSON.stringify(input);
const normalized=lowercaseChat(input.system,input.turns);
if (JSON.stringify(input)!==before || normalized.turns===input.turns || normalized.turns[0]===input.turns[0]) throw new Error('mutated display history');
console.log(JSON.stringify(normalized));'''
        proc=subprocess.run(['node','--input-type=module','-e',code,json.dumps({'system':system,'turns':turns})],cwd=root,text=True,capture_output=True,check=True)
        actual=json.loads(proc.stdout);expected_system,expected_turns=lower_chat(system,turns)
        self.assertEqual(actual,{'system':expected_system,'turns':expected_turns})

if __name__=='__main__':unittest.main()
