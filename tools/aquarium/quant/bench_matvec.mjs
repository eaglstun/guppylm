import fs from 'node:fs';
import { performance } from 'node:perf_hooks';

const source = fs.readFileSync('web/guppy.js', 'utf8');
const file = fs.readFileSync('web/model/guppy.bin');
const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);

function variant(width) {
  const vars = Array.from({length: width}, (_, i) => `a${i}`);
  const updates = vars.map((v, i) => `${v} += q[o + c + ${i}] * x[c + ${i}];`).join('\n');
  const body = `function matvec(W, x, b, y) {
    const { q, s, rows, cols } = W;
    for (let r = 0, o = 0; r < rows; r++, o += cols) {
      let ${vars.map(v => `${v} = 0`).join(', ')};
      for (let c = 0; c < cols; c += ${width}) { ${updates} }
      y[r] = (${vars.join(' + ')}) * s[r] + (b ? b[r] : 0);
    }
    return y;
  }
  `;
  const changed = source.replace(/function matvec\([\s\S]*?\n}\n\nfunction layerNorm/, `${body}\nfunction layerNorm`);
  return import(`data:text/javascript;base64,${Buffer.from(changed).toString('base64')}`);
}

function run(Guppy) {
  const m = new Guppy(buf);
  const times = [];
  let logits;
  for (let i = 0; i < 102; i++) {
    if (i === 0) times.push(performance.now());
    logits = m.step((i * 17 + 100) % 4096, i, i === 101);
  }
  times.push(performance.now());
  for (let i = 102; i < 182; i++) logits = m.step((i * 17 + 100) % 4096, i, true);
  times.push(performance.now());
  return {prompt: times[1] - times[0], generate: times[2] - times[1], checksum: logits.slice(0, 5)};
}

for (const width of [1, 2, 4, 8, 16]) {
  const { Guppy } = await variant(width);
  for (let i = 0; i < 2; i++) run(Guppy); // warm JIT
  const trials = Array.from({length: 5}, () => run(Guppy));
  const prompt = Math.min(...trials.map(x => x.prompt));
  const generate = Math.min(...trials.map(x => x.generate));
  const checksum = Array.from(trials[0].checksum).map(x => x.toFixed(3));
  console.log(JSON.stringify({width, prompt_ms: +prompt.toFixed(1), generate_ms: +generate.toFixed(1), tokens_per_second: +(80000/generate).toFixed(1), checksum}));
}
