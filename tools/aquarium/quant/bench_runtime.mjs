import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';

const history = execFileSync('git', ['log', '--format=%H', '--', 'web/guppy.js'],
  { encoding: 'utf8' }).trim().split('\n');
const original = history.map(rev => execFileSync('git', ['show', `${rev}:web/guppy.js`],
  { encoding: 'utf8' })).find(source => !source.includes('steps2('));
if (!original) throw new Error('pre-optimization web/guppy.js was not found in Git history');
const optimized = fs.readFileSync('web/guppy.js', 'utf8');
const raw = fs.readFileSync('web/model/guppy.bin');
const weights = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const load = async source => (await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)).Guppy;
const Base = await load(original), Fast = await load(optimized);
const prompt = Array.from({length: 102}, (_, i) => (i * 17 + 100) % 4096);
const continuation = Array.from({length: 80}, (_, i) => (i * 29 + 444) % 4096);

function run(Guppy) {
  const m = new Guppy(weights);
  const start = performance.now();
  let logits = m.prefill(prompt);
  const prefill = performance.now();
  const first = Float32Array.from(logits);
  for (const id of continuation) {
    logits = m.step(id, m.ids.length, true);
    m.ids.push(id);
  }
  const end = performance.now();
  return { prompt_ms: prefill - start, generation_ms: end - prefill,
           first, last: Float32Array.from(logits), kcache: m.kcache, vcache: m.vcache };
}

function maxDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

const reference = run(Base), candidate = run(Fast);
let cacheDiff = 0;
for (let i = 0; i < reference.kcache.length; i++) {
  cacheDiff = Math.max(cacheDiff, maxDiff(reference.kcache[i], candidate.kcache[i]),
                       maxDiff(reference.vcache[i], candidate.vcache[i]));
}
console.log('correctness', JSON.stringify({prompt_logits_max_abs: maxDiff(reference.first, candidate.first),
  final_logits_max_abs: maxDiff(reference.last, candidate.last), cache_max_abs: cacheDiff}));

// Check odd/even prompt lengths and prefix-cache reuse, which change pair alignment.
let variedLogits = 0, variedCache = 0;
for (const length of [1, 2, 3, 17, 18, 101, 102]) {
  for (const prior of [0, Math.floor(length / 2)]) {
    const a = new Base(weights), b = new Fast(weights), ids = prompt.slice(0, length);
    if (prior) { a.prefill(ids.slice(0, prior)); b.prefill(ids.slice(0, prior)); }
    variedLogits = Math.max(variedLogits, maxDiff(a.prefill(ids), b.prefill(ids)));
    for (let l = 0; l < a.kcache.length; l++) {
      variedCache = Math.max(variedCache, maxDiff(a.kcache[l], b.kcache[l]),
                             maxDiff(a.vcache[l], b.vcache[l]));
    }
  }
}
console.log('varied correctness', JSON.stringify({logits_max_abs: variedLogits, cache_max_abs: variedCache}));

if (process.env.ONLY_CORRECTNESS === '1') process.exit(0);

for (const [name, Guppy] of [['baseline', Base], ['optimized', Fast]]) {
  for (let i = 0; i < 2; i++) run(Guppy);
  const trials = Array.from({length: 5}, () => run(Guppy));
  const prompt = Math.min(...trials.map(x => x.prompt_ms));
  const generation = Math.min(...trials.map(x => x.generation_ms));
  console.log(name, JSON.stringify({prompt_ms: +prompt.toFixed(1), generation_ms: +generation.toFixed(1),
    generated_tokens_per_second: +(80000 / generation).toFixed(1)}));
}
