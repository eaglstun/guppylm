// GuppyLM in the browser: int8 weights, a KV cache, no dependencies.
// Mirrors guppylm/model.py (pre-norm blocks, ReLU FFN, learned positions, tied head)
// and the chat format of tools/aquarium/train_aq.py chat_ids().

const IM_START = 1, IM_END = 2;

// Copy model-facing text so the UI/history retains the visitor's original casing.
export function lowercaseChat(system, turns) {
  return { system: system.toLowerCase(),
    turns: turns.map(t => ({ ...t, content: t.content.toLowerCase() })) };
}

// ---- tokenizer: byte-level BPE, read from the Hugging Face tokenizer.json ----

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  }
  const enc = new Map(), dec = new Map();
  bs.forEach((b, i) => { enc.set(b, String.fromCodePoint(cs[i])); dec.set(String.fromCodePoint(cs[i]), b); });
  return { enc, dec };
}

const PRETOKEN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Tokenizer {
  constructor(json) {
    this.vocab = json.model.vocab;
    this.inv = [];
    for (const [s, id] of Object.entries(this.vocab)) this.inv[id] = s;
    for (const t of json.added_tokens) this.inv[t.id] = t.content;
    this.special = new Set(json.added_tokens.map(t => t.id));
    this.ranks = new Map(json.model.merges.map((m, i) => [Array.isArray(m) ? m.join(" ") : m, i]));
    const { enc, dec } = bytesToUnicode();
    this.byteEnc = enc; this.byteDec = dec;
    this.cache = new Map();
    this.utf8 = new TextEncoder();
    this.utf8dec = new TextDecoder();
  }

  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = parts[best], b = parts[best + 1], merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i++; }
        else merged.push(parts[i]);
      }
      parts = merged;
    }
    const ids = parts.map(p => this.vocab[p]);
    this.cache.set(word, ids);
    return ids;
  }

  encode(text) {
    const ids = [];
    for (const m of text.matchAll(PRETOKEN)) {
      const word = Array.from(this.utf8.encode(m[0]), b => this.byteEnc.get(b)).join("");
      ids.push(...this.bpe(word));
    }
    return ids;
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      if (this.special.has(id)) continue;
      for (const ch of this.inv[id]) bytes.push(this.byteDec.get(ch));
    }
    return this.utf8dec.decode(new Uint8Array(bytes));
  }

  chat(system, turns) {
    const nl = this.encode("\n");
    const ids = [IM_START, ...this.encode("system\n" + system), IM_END, ...nl];
    for (const t of turns) {
      ids.push(IM_START, ...this.encode(t.role + "\n"), ...this.encode(t.content), IM_END, ...nl);
    }
    ids.push(IM_START, ...this.encode("assistant\n"));
    return ids;
  }
}

// ---- model ----

function half(h) {
  const e = (h >> 10) & 31, m = h & 1023, s = h & 32768 ? -1 : 1;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function halves(buf, offset, n) {
  const h = new Uint16Array(buf, offset, n), f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = half(h[i]);
  return f;
}

// Normalized Walsh-Hadamard on each 128-wide block of n values starting at o, step apart.
function fwht(x, o, n, step) {
  for (let b = 0; b < n; b += 128) {
    for (let h = 1; h < 128; h *= 2) {
      for (let i = b; i < b + 128; i += 2 * h) {
        for (let j = i; j < i + h; j++) {
          const p = o + j * step, q = o + (j + h) * step, u = x[p], v = x[q];
          x[p] = u + v; x[q] = u - v;
        }
      }
    }
    for (let j = b; j < b + 128; j++) x[o + j * step] *= 0.08838834764831845;  // 1/sqrt(128)
  }
}

// guppy-d4-v1: every 4 weights are one of K catalog shapes (12-bit index, two per 3 bytes),
// times a row scale, in a rotated basis. Undo it once here and re-round to int8 per row,
// so the runtime below sees the same layout as guppy.bin.
function expandD4(buf, base, d, cat) {
  const [rows, cols] = d.shape, n = rows * cols / 4;
  const raw = new Uint8Array(buf, base + d.offset, Math.ceil(n / 2) * 3);
  const rs = halves(buf, base + d.scale, rows);
  const su = new Int8Array(buf, base + d.u_sign, rows), sr = new Int8Array(buf, base + d.r_sign, cols);
  const W = new Float32Array(rows * cols);
  for (let i = 0, p = 0; i < n; i += 2, p += 3) {
    const a = raw[p] | ((raw[p + 1] & 15) << 8), b = (raw[p + 1] >> 4) | (raw[p + 2] << 4);
    const s0 = rs[Math.floor(4 * i / cols)], s1 = rs[Math.floor(4 * (i + 1) / cols)];
    for (let k = 0; k < 4; k++) W[4 * i + k] = cat[4 * a + k] * s0;
    if (i + 1 < n) for (let k = 0; k < 4; k++) W[4 * i + 4 + k] = cat[4 * b + k] * s1;
  }
  for (let r = 0; r < rows; r++) fwht(W, r * cols, cols, 1);
  for (let c = 0; c < cols; c++) fwht(W, c, rows, cols);
  const q = new Int8Array(rows * cols), s = new Float32Array(rows);
  for (let r = 0, o = 0; r < rows; r++, o += cols) {
    let max = 0;
    for (let c = 0; c < cols; c++) {
      const w = (W[o + c] *= su[r] * sr[c]);
      if (Math.abs(w) > max) max = Math.abs(w);
    }
    const sc = Math.fround(max / 127) || 1;
    s[r] = sc;
    for (let c = 0; c < cols; c++) {
      const v = Math.fround(W[o + c] / sc);  // float32 division and round-half-to-even, like the exporter
      let r = Math.round(v);
      if (r - v === 0.5 && r % 2) r -= 1;
      q[o + c] = Math.max(-127, Math.min(127, r));
    }
  }
  return { q, s, rows, cols };
}

function readWeights(buf) {
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
  const base = 4 + headerLen, t = {};
  const cat = header.format === "guppy-d4-v1" ? halves(buf, base + header.catalog_offset, header.K * header.d) : null;
  for (const d of header.tensors) {
    const n = d.shape.reduce((a, b) => a * b, 1);
    if (d.dtype === "int8") {
      t[d.name] = { q: new Int8Array(buf, base + d.offset, n), s: new Float32Array(buf, base + d.scale, d.shape[0]), rows: d.shape[0], cols: d.shape[1] };
    } else if (d.dtype === "d4") {
      t[d.name] = expandD4(buf, base, d, cat);
    } else if (d.dtype === "float16") {
      t[d.name] = halves(buf, base + d.offset, n);
    } else {
      t[d.name] = new Float32Array(buf, base + d.offset, n);
    }
  }
  return { config: header.config, t };
}

// y = W x + b for an int8 matrix with a scale per output row
function matvec(W, x, b, y) {
  const { q, s, rows, cols } = W;
  for (let r = 0, o = 0; r < rows; r++, o += cols) {
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    const end = cols & ~7;
    for (let c = 0; c < end; c += 8) {
      a0 += q[o + c] * x[c];       a1 += q[o + c + 1] * x[c + 1];
      a2 += q[o + c + 2] * x[c + 2]; a3 += q[o + c + 3] * x[c + 3];
      a4 += q[o + c + 4] * x[c + 4]; a5 += q[o + c + 5] * x[c + 5];
      a6 += q[o + c + 6] * x[c + 6]; a7 += q[o + c + 7] * x[c + 7];
    }
    for (let c = end; c < cols; c++) a0 += q[o + c] * x[c];
    y[r] = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7) * s[r] + (b ? b[r] : 0);
  }
  return y;
}

// Two prompt tokens share each weight read. Their attention remains causal.
function matvec2(W, x0, x1, b, y0, y1) {
  const { q, s, rows, cols } = W;
  for (let r = 0, o = 0; r < rows; r++, o += cols) {
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    let z0 = 0, z1 = 0, z2 = 0, z3 = 0;
    const end = cols & ~3;
    for (let c = 0; c < end; c += 4) {
      const q0 = q[o + c], q1 = q[o + c + 1], q2 = q[o + c + 2], q3 = q[o + c + 3];
      a0 += q0 * x0[c];     z0 += q0 * x1[c];
      a1 += q1 * x0[c + 1]; z1 += q1 * x1[c + 1];
      a2 += q2 * x0[c + 2]; z2 += q2 * x1[c + 2];
      a3 += q3 * x0[c + 3]; z3 += q3 * x1[c + 3];
    }
    for (let c = end; c < cols; c++) {
      const v = q[o + c]; a0 += v * x0[c]; z0 += v * x1[c];
    }
    const bias = b ? b[r] : 0, scale = s[r];
    y0[r] = (a0 + a1 + a2 + a3) * scale + bias;
    y1[r] = (z0 + z1 + z2 + z3) * scale + bias;
  }
}

function layerNorm(x, g, b, y) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
  return y;
}

export class Guppy {
  constructor(buf) {
    const { config, t } = readWeights(buf);
    this.c = config; this.t = t;
    const d = config.d_model, L = config.n_layers, T = config.max_seq_len;
    this.hd = d / config.n_heads;
    this.kcache = Array.from({ length: L }, () => new Float32Array(T * d));
    this.vcache = Array.from({ length: L }, () => new Float32Array(T * d));
    this.x = new Float32Array(d); this.h = new Float32Array(d); this.a = new Float32Array(d);
    this.qkv = new Float32Array(3 * d); this.f = new Float32Array(config.ffn_hidden);
    this.pair = {
      x0: new Float32Array(d), x1: new Float32Array(d),
      h0: new Float32Array(d), h1: new Float32Array(d),
      a0: new Float32Array(d), a1: new Float32Array(d),
      q0: new Float32Array(3 * d), q1: new Float32Array(3 * d),
      f0: new Float32Array(config.ffn_hidden), f1: new Float32Array(config.ffn_hidden),
    };
    this.att = new Float32Array(T); this.logits = new Float32Array(config.vocab_size);
    this.ids = [];  // tokens currently in the cache
  }

  // Run one token at position pos; returns logits only when asked (the head is 4096 x 384).
  step(id, pos, wantLogits) {
    const { c, t, x, h, a, qkv, f, att, hd } = this, d = c.d_model;
    const te = t["tok_emb.weight"], pe = t["pos_emb.weight"];
    for (let i = 0; i < d; i++) x[i] = te.q[id * d + i] * te.s[id] + pe.q[pos * d + i] * pe.s[pos];
    for (let l = 0; l < c.n_layers; l++) {
      const p = `blocks.${l}.`;
      layerNorm(x, t[p + "norm1.weight"], t[p + "norm1.bias"], h);
      matvec(t[p + "attn.qkv.weight"], h, t[p + "attn.qkv.bias"], qkv);
      const K = this.kcache[l], V = this.vcache[l];
      K.set(qkv.subarray(d, 2 * d), pos * d);
      V.set(qkv.subarray(2 * d, 3 * d), pos * d);
      const scale = 1 / Math.sqrt(hd);
      for (let head = 0; head < c.n_heads; head++) {
        const o = head * hd;
        let max = -Infinity;
        for (let j = 0; j <= pos; j++) {
          let s = 0;
          for (let k = 0; k < hd; k++) s += qkv[o + k] * K[j * d + o + k];
          att[j] = s * scale;
          if (att[j] > max) max = att[j];
        }
        let sum = 0;
        for (let j = 0; j <= pos; j++) { att[j] = Math.exp(att[j] - max); sum += att[j]; }
        for (let k = 0; k < hd; k++) h[o + k] = 0;
        for (let j = 0; j <= pos; j++) {
          const w = att[j] / sum;
          for (let k = 0; k < hd; k++) h[o + k] += w * V[j * d + o + k];
        }
      }
      matvec(t[p + "attn.out.weight"], h, t[p + "attn.out.bias"], a);
      for (let i = 0; i < d; i++) x[i] += a[i];
      layerNorm(x, t[p + "norm2.weight"], t[p + "norm2.bias"], h);
      matvec(t[p + "ffn.up.weight"], h, t[p + "ffn.up.bias"], f);
      for (let i = 0; i < f.length; i++) if (f[i] < 0) f[i] = 0;
      matvec(t[p + "ffn.down.weight"], f, t[p + "ffn.down.bias"], a);
      for (let i = 0; i < d; i++) x[i] += a[i];
    }
    if (!wantLogits) return null;
    layerNorm(x, t["norm.weight"], t["norm.bias"], h);
    return matvec(te, h, null, this.logits);
  }

  attend(l, qkv, pos, h) {
    const { c, hd, att } = this, d = c.d_model;
    const K = this.kcache[l], V = this.vcache[l];
    const scale = 1 / Math.sqrt(hd);
    for (let head = 0; head < c.n_heads; head++) {
      const o = head * hd;
      let max = -Infinity;
      for (let j = 0; j <= pos; j++) {
        let s = 0;
        for (let k = 0; k < hd; k++) s += qkv[o + k] * K[j * d + o + k];
        att[j] = s * scale;
        if (att[j] > max) max = att[j];
      }
      let sum = 0;
      for (let j = 0; j <= pos; j++) { att[j] = Math.exp(att[j] - max); sum += att[j]; }
      for (let k = 0; k < hd; k++) h[o + k] = 0;
      for (let j = 0; j <= pos; j++) {
        const w = att[j] / sum;
        for (let k = 0; k < hd; k++) h[o + k] += w * V[j * d + o + k];
      }
    }
  }

  steps2(id0, id1, pos, wantLogits) {
    const { c, t } = this, d = c.d_model;
    const te = t["tok_emb.weight"], pe = t["pos_emb.weight"];
    const { x0, x1, h0, h1, a0, a1, q0, q1, f0, f1 } = this.pair;
    for (let i = 0; i < d; i++) {
      x0[i] = te.q[id0 * d + i] * te.s[id0] + pe.q[pos * d + i] * pe.s[pos];
      x1[i] = te.q[id1 * d + i] * te.s[id1] + pe.q[(pos + 1) * d + i] * pe.s[pos + 1];
    }
    for (let l = 0; l < c.n_layers; l++) {
      const p = `blocks.${l}.`;
      layerNorm(x0, t[p + "norm1.weight"], t[p + "norm1.bias"], h0);
      layerNorm(x1, t[p + "norm1.weight"], t[p + "norm1.bias"], h1);
      matvec2(t[p + "attn.qkv.weight"], h0, h1, t[p + "attn.qkv.bias"], q0, q1);
      const K = this.kcache[l], V = this.vcache[l];
      K.set(q0.subarray(d, 2 * d), pos * d);
      V.set(q0.subarray(2 * d, 3 * d), pos * d);
      K.set(q1.subarray(d, 2 * d), (pos + 1) * d);
      V.set(q1.subarray(2 * d, 3 * d), (pos + 1) * d);
      this.attend(l, q0, pos, h0);
      this.attend(l, q1, pos + 1, h1);
      matvec2(t[p + "attn.out.weight"], h0, h1, t[p + "attn.out.bias"], a0, a1);
      for (let i = 0; i < d; i++) { x0[i] += a0[i]; x1[i] += a1[i]; }
      layerNorm(x0, t[p + "norm2.weight"], t[p + "norm2.bias"], h0);
      layerNorm(x1, t[p + "norm2.weight"], t[p + "norm2.bias"], h1);
      matvec2(t[p + "ffn.up.weight"], h0, h1, t[p + "ffn.up.bias"], f0, f1);
      for (let i = 0; i < f0.length; i++) { if (f0[i] < 0) f0[i] = 0; if (f1[i] < 0) f1[i] = 0; }
      matvec2(t[p + "ffn.down.weight"], f0, f1, t[p + "ffn.down.bias"], a0, a1);
      for (let i = 0; i < d; i++) { x0[i] += a0[i]; x1[i] += a1[i]; }
    }
    if (!wantLogits) return null;
    layerNorm(x1, t["norm.weight"], t["norm.bias"], h1);
    return matvec(te, h1, null, this.logits);
  }

  // Feed a prompt, reusing whatever prefix is already in the cache.
  prefill(ids) {
    let keep = 0;
    while (keep < this.ids.length && keep < ids.length - 1 && this.ids[keep] === ids[keep]) keep++;
    this.ids = ids.slice(0, keep);
    let logits = null;
    let i = keep;
    for (; i + 1 < ids.length; i += 2) {
      logits = this.steps2(ids[i], ids[i + 1], i, i + 1 === ids.length - 1);
      this.ids.push(ids[i], ids[i + 1]);
    }
    for (; i < ids.length; i++) {
      logits = this.step(ids[i], i, i === ids.length - 1);
      this.ids.push(ids[i]);
    }
    return logits;
  }

  *generate(prompt, { maxNew = 80, temperature = 0.7, topK = 40, random = Math.random } = {}) {
    const room = this.c.max_seq_len - maxNew;
    if (prompt.length > room) prompt = prompt.slice(prompt.length - room);
    let logits = this.prefill(prompt);
    for (let n = 0; n < maxNew; n++) {
      const id = temperature > 0 ? sampleTopK(logits, temperature, topK, random) : argmax(logits);
      if (id === IM_END) return;
      yield id;
      logits = this.step(id, this.ids.length, true);
      this.ids.push(id);
    }
  }
}

function argmax(v) {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

function sampleTopK(logits, temperature, k, random) {
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  const max = logits[idx[0]] / temperature;
  const p = idx.map(i => Math.exp(logits[i] / temperature - max));
  let r = random() * p.reduce((a, b) => a + b, 0);
  for (let i = 0; i < k; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
  return idx[k - 1];
}
