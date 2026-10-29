// Exercise the actual worker source over local HTTP, with browser globals adapted for Node.
import fs from 'node:fs';
import assert from 'node:assert/strict';
const base='http://127.0.0.1:8766/';const nativeFetch=globalThis.fetch;
globalThis.fetch=url=>nativeFetch(new URL(url,base));globalThis.onmessage=null;
let ready,done;const readyP=new Promise(r=>ready=r),doneP=new Promise(r=>done=r);
globalThis.postMessage=m=>{if(m.type==='ready')ready(m);if(m.type==='done')done(m);if(m.type==='error')throw new Error(m.message);};
const guppy='data:text/javascript;base64,'+fs.readFileSync('web/guppy.js').toString('base64');
const source=fs.readFileSync('web/worker.js','utf8').replace(/from "\.\/guppy\.js[^\"]*"/,`from "${guppy}"`);
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const loaded=await readyP;assert.equal(loaded.bytes,fs.statSync('web/model/guppy.bin').size);
const data={type:'chat',system:'You are Pebble, a small guppy. You are friendly and curious. You live in an aquarium.',turns:[{role:'user',content:'LOOK! I MADE A LITTLE BRIDGE!'}],temperature:0};
const before=JSON.stringify(data);globalThis.onmessage({data});const reply=await doneP;
assert.equal(JSON.stringify(data),before);assert.ok(reply.text.length>0);assert.equal(reply.text,reply.text.toLowerCase());
const result={loaded_bytes:loaded.bytes,original_user_text_preserved:true,response:reply.text,tokens:reply.tokens,passed:true};
fs.writeFileSync('experiments/v52/results/lowercase-web/worker-smoke.json',JSON.stringify(result,null,2)+'\n');console.log(result);
