// Experimental direct-code Guppy runtime. Weights stay as 16-bit codes for
// eight values; a 512 KiB table expands one code inside each matrix multiply.
import { Guppy, Tokenizer } from './guppy.js';
export { Tokenizer };

function half(h) {
  const sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 31, frac = h & 1023;
  if (exp === 0) return sign * Math.pow(2, -14) * frac / 1024;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function readHalf(buf, offset, n) {
  const data = new DataView(buf, offset, n * 2), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = half(data.getUint16(i * 2, true));
  return out;
}

function makeLookup(grid) {
  const lut = new Int8Array(65536 * 8);
  for (let code = 0; code < 65536; code++) {
    const raw = code & 255, shape = code >>> 8;
    let parity = raw;
    parity ^= parity >>> 4; parity ^= parity >>> 2; parity ^= parity >>> 1;
    parity &= 1;
    const flips = raw ^ parity, shift = parity ? -1 : 1;
    for (let i = 0; i < 8; i++) {
      const packed = grid[shape * 4 + (i >>> 1)];
      const nib = i & 1 ? packed >>> 4 : packed & 15;
      const baseTimes2 = nib - 8;
      lut[code * 8 + i] = 2 * baseTimes2 * ((flips >>> i) & 1 ? -1 : 1) + shift;
    }
  }
  return lut;
}

function readWeights(buf) {
  const length = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, length)));
  if (header.format !== 'guppy-e8-v1' || header.rotation_block !== 128) throw new Error('unsupported E8 format');
  const base = 4 + length, t = {};
  for (const d of header.tensors) {
    const n = d.shape.reduce((a, b) => a * b, 1);
    if (d.dtype === 'e8') {
      const [rows, cols] = d.shape;
      const groups = cols / d.scale_group;
      t[d.name] = { rows, cols,
        codes: new Uint16Array(buf, base + d.offset, n / 8),
        scale: readHalf(buf, base + d.scale, rows * groups),
        groups, groupChunks: d.scale_group / 8,
        uSign: new Int8Array(buf, base + d.u_sign, rows),
        rSign: new Int8Array(buf, base + d.r_sign, cols),
        inputRot: new Float32Array(cols), outputRot: new Float32Array(rows),
        inputRot2: new Float32Array(cols), outputRot2: new Float32Array(rows) };
    } else if (d.dtype === 'int8') {
      t[d.name] = { q: new Int8Array(buf, base + d.offset, n),
        scale: new Float32Array(buf, base + d.scale, d.shape[0]),
        rows: d.shape[0], cols: d.shape[1] };
    } else if (d.dtype === 'float16') {
      t[d.name] = readHalf(buf, base + d.offset, n);
    } else throw new Error(`unknown tensor dtype ${d.dtype}`);
  }
  const grid = new Uint8Array(buf, base + header.base_grid_offset, 1024);
  return { c: header.config, t, lut: makeLookup(grid) };
}

function fwht128(a, start) {
  for (let width = 1; width < 128; width *= 2)
    for (let block = start; block < start + 128; block += width * 2)
      for (let i = 0; i < width; i++) {
        const p = block + i, x = a[p], y = a[p + width];
        a[p] = x + y; a[p + width] = x - y;
      }
}

function rotateInput(x, out, signs) { // R^T x = H diag(signs) x
  for (let start = 0; start < x.length; start += 128) {
    for (let i = 0; i < 128; i++) out[start + i] = x[start + i] * signs[start + i];
    fwht128(out, start);
    for (let i = 0; i < 128; i++) out[start + i] *= 1 / Math.sqrt(128);
  }
}

function rotateOutput(x, out, signs) { // U x = diag(signs) H x
  out.set(x);
  for (let start = 0; start < x.length; start += 128) {
    fwht128(out, start);
    for (let i = 0; i < 128; i++) out[start + i] *= signs[start + i] / Math.sqrt(128);
  }
}

function matvec(W, x, b, y, lut) {
  if (W.groups > 1) return matvecGroups(W, x, b, y, lut);
  rotateInput(x, W.inputRot, W.rSign);
  const v = W.inputRot, out = W.outputRot, chunks = W.cols / 8;
  for (let r = 0, o = 0; r < W.rows; r++, o += chunks) {
    let a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (let c = 0; c < chunks; c++) {
      const p = W.codes[o + c] * 8, j = c * 8;
      a0 += lut[p] * v[j]; a1 += lut[p+1] * v[j+1];
      a2 += lut[p+2] * v[j+2]; a3 += lut[p+3] * v[j+3];
      a4 += lut[p+4] * v[j+4]; a5 += lut[p+5] * v[j+5];
      a6 += lut[p+6] * v[j+6]; a7 += lut[p+7] * v[j+7];
    }
    out[r] = (a0+a1+a2+a3+a4+a5+a6+a7) * W.scale[r] / 4;
  }
  rotateOutput(out, y, W.uSign);
  if (b) for (let r = 0; r < W.rows; r++) y[r] += b[r];
  return y;
}

function matvecGroups(W, x, b, y, lut) {
  rotateInput(x, W.inputRot, W.rSign);
  const v = W.inputRot, out = W.outputRot, chunks = W.cols / 8;
  for (let r = 0, o = 0; r < W.rows; r++, o += chunks) {
    let total = 0;
    for (let g = 0; g < W.groups; g++) {
      let a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
      for (let c = g * W.groupChunks; c < (g + 1) * W.groupChunks; c++) {
        const p = W.codes[o+c]*8, j = c*8;
        a0+=lut[p]*v[j];a1+=lut[p+1]*v[j+1];a2+=lut[p+2]*v[j+2];a3+=lut[p+3]*v[j+3];
        a4+=lut[p+4]*v[j+4];a5+=lut[p+5]*v[j+5];a6+=lut[p+6]*v[j+6];a7+=lut[p+7]*v[j+7];
      }
      total += (a0+a1+a2+a3+a4+a5+a6+a7) * W.scale[r*W.groups+g] / 4;
    }
    out[r] = total;
  }
  rotateOutput(out, y, W.uSign);
  if (b) for (let r = 0; r < W.rows; r++) y[r] += b[r];
  return y;
}

function matvec2(W, x0, x1, b, y0, y1, lut) {
  if (W.groups > 1) return matvec2Groups(W, x0, x1, b, y0, y1, lut);
  rotateInput(x0, W.inputRot, W.rSign);
  rotateInput(x1, W.inputRot2, W.rSign);
  const v0 = W.inputRot, v1 = W.inputRot2;
  const o0 = W.outputRot, o1 = W.outputRot2, chunks = W.cols / 8;
  for (let r = 0, o = 0; r < W.rows; r++, o += chunks) {
    let a0=0,a1=0,a2=0,a3=0;
    let z0=0,z1=0,z2=0,z3=0;
    for (let c = 0; c < chunks; c++) {
      const p = W.codes[o + c] * 8, j = c * 8;
      const q0=lut[p],q1=lut[p+1],q2=lut[p+2],q3=lut[p+3];
      const q4=lut[p+4],q5=lut[p+5],q6=lut[p+6],q7=lut[p+7];
      a0+=q0*v0[j]+q4*v0[j+4];a1+=q1*v0[j+1]+q5*v0[j+5];
      a2+=q2*v0[j+2]+q6*v0[j+6];a3+=q3*v0[j+3]+q7*v0[j+7];
      z0+=q0*v1[j]+q4*v1[j+4];z1+=q1*v1[j+1]+q5*v1[j+5];
      z2+=q2*v1[j+2]+q6*v1[j+6];z3+=q3*v1[j+3]+q7*v1[j+7];
    }
    const scale = W.scale[r] / 4;
    o0[r] = (a0+a1+a2+a3) * scale;
    o1[r] = (z0+z1+z2+z3) * scale;
  }
  rotateOutput(o0, y0, W.uSign);
  rotateOutput(o1, y1, W.uSign);
  if (b) for (let r = 0; r < W.rows; r++) { y0[r] += b[r]; y1[r] += b[r]; }
}

function matvec2Groups(W, x0, x1, b, y0, y1, lut) {
  rotateInput(x0, W.inputRot, W.rSign);
  rotateInput(x1, W.inputRot2, W.rSign);
  const v0=W.inputRot,v1=W.inputRot2,o0=W.outputRot,o1=W.outputRot2,chunks=W.cols/8;
  for(let r=0,o=0;r<W.rows;r++,o+=chunks){
    let total0=0,total1=0;
    for(let g=0;g<W.groups;g++){
      let a0=0,a1=0,a2=0,a3=0,z0=0,z1=0,z2=0,z3=0;
      for(let c=g*W.groupChunks;c<(g+1)*W.groupChunks;c++){
        const p=W.codes[o+c]*8,j=c*8;
        const q0=lut[p],q1=lut[p+1],q2=lut[p+2],q3=lut[p+3];
        const q4=lut[p+4],q5=lut[p+5],q6=lut[p+6],q7=lut[p+7];
        a0+=q0*v0[j]+q4*v0[j+4];a1+=q1*v0[j+1]+q5*v0[j+5];
        a2+=q2*v0[j+2]+q6*v0[j+6];a3+=q3*v0[j+3]+q7*v0[j+7];
        z0+=q0*v1[j]+q4*v1[j+4];z1+=q1*v1[j+1]+q5*v1[j+5];
        z2+=q2*v1[j+2]+q6*v1[j+6];z3+=q3*v1[j+3]+q7*v1[j+7];
      }
      const scale=W.scale[r*W.groups+g]/4;
      total0+=(a0+a1+a2+a3)*scale;total1+=(z0+z1+z2+z3)*scale;
    }
    o0[r]=total0;o1[r]=total1;
  }
  rotateOutput(o0,y0,W.uSign);rotateOutput(o1,y1,W.uSign);
  if(b)for(let r=0;r<W.rows;r++){y0[r]+=b[r];y1[r]+=b[r];}
}

function norm(x, g, b, y) {
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

function hadamardSign(i,j) {
  let a = i & j; a ^= a >>> 4; a ^= a >>> 2; a ^= a >>> 1;
  return a & 1 ? -1 : 1;
}

export class GuppyE8 {
  constructor(buf) {
    const { c, t, lut } = readWeights(buf);
    this.c = c; this.t = t; this.lut = lut; this.buf = buf;
    const d = c.d_model, L = c.n_layers, T = c.max_seq_len;
    this.hd = d / c.n_heads;
    this.kcache = Array.from({ length: L }, () => new Float32Array(T * d));
    this.vcache = Array.from({ length: L }, () => new Float32Array(T * d));
    this.x = new Float32Array(d); this.h = new Float32Array(d); this.a = new Float32Array(d);
    this.qkv = new Float32Array(3 * d); this.f = new Float32Array(c.ffn_hidden);
    this.pair = {x0:new Float32Array(d),x1:new Float32Array(d),
      h0:new Float32Array(d),h1:new Float32Array(d),a0:new Float32Array(d),a1:new Float32Array(d),
      q0:new Float32Array(3*d),q1:new Float32Array(3*d),
      f0:new Float32Array(c.ffn_hidden),f1:new Float32Array(c.ffn_hidden)};
    this.att = new Float32Array(T); this.logits = new Float32Array(c.vocab_size);
    this.ids = []; this.embeddingCache = new Map();
  }

  embedding(id) {
    if (this.embeddingCache.has(id)) return this.embeddingCache.get(id);
    const W = this.t['tok_emb.weight'], d = W.cols, chunks = d / 8;
    const acc = new Float32Array(d), out = new Float32Array(d);
    const first = id & ~127, local = id & 127;
    for (let j = 0; j < 128; j++) {
      const row = first + j, o = row * chunks;
      const coeff = W.uSign[id] * hadamardSign(local,j) / (4 * Math.sqrt(128));
      for (let c = 0; c < chunks; c++) {
        const p = W.codes[o + c] * 8, q = c * 8;
        const scale = W.scale[row*W.groups + Math.floor(c/W.groupChunks)];
        for (let z = 0; z < 8; z++) acc[q+z] += coeff * scale * this.lut[p+z];
      }
    }
    // A row of U Q R^T is a column of R applied to the mixed Q row.
    rotateOutput(acc, out, W.rSign);
    this.embeddingCache.set(id,out);
    return out;
  }

  step(id, pos, wantLogits) {
    const { c, t, x, h, a, qkv, f, lut } = this, d = c.d_model;
    const te = t['tok_emb.weight'], pe = t['pos_emb.weight'];
    const row = this.embedding(id);
    for (let i = 0; i < d; i++) x[i] = row[i] + pe.q[pos*d+i] * pe.scale[pos];
    for (let l = 0; l < c.n_layers; l++) {
      const p = `blocks.${l}.`;
      norm(x, t[p+'norm1.weight'], t[p+'norm1.bias'], h);
      matvec(t[p+'attn.qkv.weight'], h, t[p+'attn.qkv.bias'], qkv, lut);
      const K = this.kcache[l], V = this.vcache[l];
      K.set(qkv.subarray(d, 2*d), pos*d);
      V.set(qkv.subarray(2*d, 3*d), pos*d);
      Guppy.prototype.attend.call(this, l, qkv, pos, h);
      matvec(t[p+'attn.out.weight'], h, t[p+'attn.out.bias'], a, lut);
      for (let i = 0; i < d; i++) x[i] += a[i];
      norm(x, t[p+'norm2.weight'], t[p+'norm2.bias'], h);
      matvec(t[p+'ffn.up.weight'], h, t[p+'ffn.up.bias'], f, lut);
      for (let i = 0; i < f.length; i++) if (f[i] < 0) f[i] = 0;
      matvec(t[p+'ffn.down.weight'], f, t[p+'ffn.down.bias'], a, lut);
      for (let i = 0; i < d; i++) x[i] += a[i];
    }
    if (!wantLogits) return null;
    norm(x, t['norm.weight'], t['norm.bias'], h);
    return matvec(te, h, null, this.logits, lut);
  }

  steps2(id0,id1,pos,wantLogits) {
    const { c, t, lut } = this, d = c.d_model;
    const pe = t['pos_emb.weight'], te = t['tok_emb.weight'];
    const {x0,x1,h0,h1,a0,a1,q0,q1,f0,f1} = this.pair;
    const e0=this.embedding(id0), e1=this.embedding(id1);
    for(let i=0;i<d;i++){
      x0[i]=e0[i]+pe.q[pos*d+i]*pe.scale[pos];
      x1[i]=e1[i]+pe.q[(pos+1)*d+i]*pe.scale[pos+1];
    }
    for(let l=0;l<c.n_layers;l++){
      const p=`blocks.${l}.`;
      norm(x0,t[p+'norm1.weight'],t[p+'norm1.bias'],h0);
      norm(x1,t[p+'norm1.weight'],t[p+'norm1.bias'],h1);
      matvec2(t[p+'attn.qkv.weight'],h0,h1,t[p+'attn.qkv.bias'],q0,q1,lut);
      const K=this.kcache[l],V=this.vcache[l];
      K.set(q0.subarray(d,2*d),pos*d);V.set(q0.subarray(2*d,3*d),pos*d);
      K.set(q1.subarray(d,2*d),(pos+1)*d);V.set(q1.subarray(2*d,3*d),(pos+1)*d);
      Guppy.prototype.attend.call(this,l,q0,pos,h0);
      Guppy.prototype.attend.call(this,l,q1,pos+1,h1);
      matvec2(t[p+'attn.out.weight'],h0,h1,t[p+'attn.out.bias'],a0,a1,lut);
      for(let i=0;i<d;i++){x0[i]+=a0[i];x1[i]+=a1[i];}
      norm(x0,t[p+'norm2.weight'],t[p+'norm2.bias'],h0);
      norm(x1,t[p+'norm2.weight'],t[p+'norm2.bias'],h1);
      matvec2(t[p+'ffn.up.weight'],h0,h1,t[p+'ffn.up.bias'],f0,f1,lut);
      for(let i=0;i<f0.length;i++){if(f0[i]<0)f0[i]=0;if(f1[i]<0)f1[i]=0;}
      matvec2(t[p+'ffn.down.weight'],f0,f1,t[p+'ffn.down.bias'],a0,a1,lut);
      for(let i=0;i<d;i++){x0[i]+=a0[i];x1[i]+=a1[i];}
    }
    if(!wantLogits)return null;
    norm(x1,t['norm.weight'],t['norm.bias'],h1);
    return matvec(te,h1,null,this.logits,lut);
  }
  prefill(ids) { return Guppy.prototype.prefill.call(this, ids); }
  *generate(prompt, opts) { yield* Guppy.prototype.generate.call(this, prompt, opts); }
  residentWeightBytes() {
    let small = 0, scratch = 0;
    for (const W of Object.values(this.t)) {
      if (W instanceof Float32Array) small += W.byteLength;
      else if (W.codes) scratch += W.inputRot.byteLength + W.outputRot.byteLength +
        W.inputRot2.byteLength + W.outputRot2.byteLength + W.scale.byteLength;
    }
    for (const v of this.embeddingCache.values()) small += v.byteLength;
    return this.buf.byteLength + this.lut.byteLength + small + scratch;
  }
}
