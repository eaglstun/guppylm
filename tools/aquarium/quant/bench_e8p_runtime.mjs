import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { Guppy } from '../../../web/guppy.js';
import { GuppyE8 } from '../../../web/guppy_e8p.js';

function arrayBuffer(path) {
  const b = fs.readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

const compressed = arrayBuffer(process.env.E8_BIN || 'web/model/guppy_e8.bin');
const expandedInt8 = arrayBuffer(process.env.E8_INT8 || 'runs/quant-e8p/int8ref/guppy.bin');
const reference = JSON.parse(fs.readFileSync(process.env.E8_REF || 'runs/quant-e8p/web_reference.json', 'utf8'));

function maxDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

function memory(m, buf) {
  const weightStorage = typeof m.residentWeightBytes === 'function' ? m.residentWeightBytes() : buf.byteLength;
  const kv = [...m.kcache,...m.vcache].reduce((n,a)=>n+a.byteLength,0);
  const activations = [m.x,m.h,m.a,m.qkv,m.f,m.att,m.logits]
    .reduce((n,a)=>n+a.byteLength,0);
  const pairedScratch = m.pair ? Object.values(m.pair).reduce((n,a)=>n+a.byteLength,0) : 0;
  return {weight_storage_bytes:weightStorage,kv_cache_bytes:kv,
    activation_bytes:activations+pairedScratch,
    total_counted_bytes:weightStorage+kv+activations+pairedScratch,
    cached_embedding_rows:m.embeddingCache?.size ?? 0};
}

function check(Model, buf) {
  const m = new Model(buf), logits = [];
  for (let i = 0; i < reference.ids.length; i++)
    logits.push(Float32Array.from(m.step(reference.ids[i], i, true)));
  const p = new Model(buf), prefill = Float32Array.from(p.prefill(reference.ids));
  const d = m.c.d_model, count = reference.ids.length;
  return { logits: Math.max(...logits.map((v,i)=>maxDiff(v,reference.logits[i]))),
    prefill_vs_step: maxDiff(prefill, logits.at(-1)),
    k: Math.max(...m.kcache.map((v,l)=>maxDiff(v.subarray(0,count*d),reference.k[l]))),
    v: Math.max(...m.vcache.map((v,l)=>maxDiff(v.subarray(0,count*d),reference.v[l]))),
    memory:memory(m,buf) };
}

console.log('compressed versus Python', JSON.stringify(check(GuppyE8,compressed)));
console.log('int8 versus Python', JSON.stringify(check(Guppy,expandedInt8)));

if (process.env.ONLY_CORRECTNESS === '1') process.exit(0);

const ids = Array.from({length:182},(_,i)=>(100+i*17)%4096);
function run(Model, buf) {
  const m = new Model(buf);
  let t = performance.now();
  m.prefill(ids.slice(0,102));
  const prompt = performance.now() - t;
  t = performance.now();
  for (let i = 102; i < ids.length; i++) m.step(ids[i],i,true);
  const generation = performance.now() - t;
  return {prompt,generation,memory:memory(m,buf)};
}
for (const [name,Model,buf] of [['int8',Guppy,expandedInt8],['compressed',GuppyE8,compressed]]) {
  run(Model,buf); run(Model,buf);
  const trials = Array.from({length:5},()=>run(Model,buf));
  const p = Math.min(...trials.map(t=>t.prompt)), g = Math.min(...trials.map(t=>t.generation));
  console.log('timing',JSON.stringify({name,prompt_ms:+p.toFixed(1),generation_ms:+g.toFixed(1),tokens_per_second:+(80000/g).toFixed(1),
    after_182_tokens:trials[0].memory}));
}
