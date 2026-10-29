import fs from 'node:fs';
import assert from 'node:assert/strict';
const {Tokenizer,Guppy,lowercaseChat}=await import('data:text/javascript;base64,'+fs.readFileSync('web/guppy.js').toString('base64'));
const ref=JSON.parse(fs.readFileSync('runs/v52/lowercase-web/reference.json','utf8'));
const tok=new Tokenizer(JSON.parse(fs.readFileSync('web/model/tokenizer.json','utf8')));
const b=fs.readFileSync('web/model/guppy.bin');const weights=b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);
const cases=[];
for(const test of ref.cases){
 const before=JSON.stringify(test.turns);const input=lowercaseChat(test.system,test.turns);const ids=tok.chat(input.system,input.turns);
 assert.equal(JSON.stringify(test.turns),before);assert.deepEqual(ids,test.ids);
 const model=new Guppy(weights);const logits=model.prefill(ids);
 const maxError=Math.max(...logits.map((v,i)=>Math.abs(v-test.first_logits[i])));
 assert.ok(maxError<1e-3,`logit error ${maxError}`);
 const generated=[...model.generate(ids,{temperature:0,maxNew:24})];assert.deepEqual(generated,test.greedy_tokens);
 cases.push({prompt:test.turns[0].content,prompt_tokens:ids.length,greedy_tokens:generated.length,max_logit_error:maxError,response:tok.decode(generated)});
}
const report={binary_sha256:ref.binary_sha256,checkpoint_sha256:ref.checkpoint_sha256,cases,passed:true};
fs.writeFileSync('experiments/v52/results/lowercase-web/parity.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
