// Synthetic microbenchmark of resident int8 weights versus direct 16-bit/8-weight codes.
// The same decoded matrix values are used in both paths; this isolates lookup cost.
import { performance } from 'node:perf_hooks';

const dims = [[4096,384], ...Array.from({length:6}, () => [[1152,384],[384,384],[768,384],[384,768]]).flat()];
const lut = new Int8Array(65536 * 8);
for (let i = 0; i < lut.length; i++) lut[i] = (i * 37 + (i >> 8)) % 15 - 7;

function int8mat(W, x, y) {
  const { q, scale, rows, cols } = W;
  for (let r = 0, o = 0; r < rows; r++, o += cols) {
    let a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (let c=0;c<cols;c+=8) {
      a0+=q[o+c]*x[c]; a1+=q[o+c+1]*x[c+1];
      a2+=q[o+c+2]*x[c+2]; a3+=q[o+c+3]*x[c+3];
      a4+=q[o+c+4]*x[c+4]; a5+=q[o+c+5]*x[c+5];
      a6+=q[o+c+6]*x[c+6]; a7+=q[o+c+7]*x[c+7];
    }
    y[r]=(a0+a1+a2+a3+a4+a5+a6+a7)*scale[r];
  }
}

function codedmat(W, x, y) {
  const { codes, scale, rows, cols } = W, chunks=cols/8;
  for (let r=0, o=0;r<rows;r++,o+=chunks) {
    let a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (let c=0;c<chunks;c++) {
      const p=codes[o+c]*8, j=c*8;
      a0+=lut[p]*x[j]; a1+=lut[p+1]*x[j+1];
      a2+=lut[p+2]*x[j+2]; a3+=lut[p+3]*x[j+3];
      a4+=lut[p+4]*x[j+4]; a5+=lut[p+5]*x[j+5];
      a6+=lut[p+6]*x[j+6]; a7+=lut[p+7]*x[j+7];
    }
    y[r]=(a0+a1+a2+a3+a4+a5+a6+a7)*scale[r];
  }
}

function hadamard128(src,dst,signs,signsBefore){
  for(let off=0;off<src.length;off+=128){
    for(let i=0;i<128;i++)dst[off+i]=src[off+i]*(signsBefore?signs[off+i]:1);
    for(let span=1;span<128;span*=2)
      for(let block=off;block<off+128;block+=span*2)
        for(let j=0;j<span;j++){
          const a=dst[block+j],b=dst[block+j+span];
          dst[block+j]=a+b;dst[block+j+span]=a-b;
        }
    for(let i=0;i<128;i++)dst[off+i]*=(signsBefore?1:signs[off+i])/Math.sqrt(128);
  }
}

function codedrot(W,x,y){
  hadamard128(x,W.inputRot,W.inputSigns,true);
  codedmat(W,W.inputRot,W.outputRot);
  hadamard128(W.outputRot,y,W.outputSigns,false);
}

const matrices=dims.map(([rows,cols],n)=>{
  const codes=new Uint16Array(rows*cols/8), q=new Int8Array(rows*cols);
  for(let i=0;i<codes.length;i++){
    const code=(i*977+n*131)&65535; codes[i]=code;
    q.set(lut.subarray(code*8,code*8+8),i*8);
  }
  const scale=new Float32Array(rows).fill(1/4);
  const x=new Float32Array(cols);
  for(let i=0;i<cols;i++)x[i]=Math.sin(i*.013);
  const inputSigns=new Int8Array(cols),outputSigns=new Int8Array(rows);
  for(let i=0;i<cols;i++)inputSigns[i]=i%3?1:-1;
  for(let i=0;i<rows;i++)outputSigns[i]=i%5?1:-1;
  return {W:{q,codes,scale,rows,cols,inputSigns,outputSigns,inputRot:new Float32Array(cols),outputRot:new Float32Array(rows)},x,y:new Float32Array(rows)};
});

function sweep(fn){for(const {W,x,y} of matrices) fn(W,x,y);}
for(let i=0;i<5;i++){sweep(int8mat);sweep(codedmat);}
const check=[];
for(const {W,x,y} of matrices){
  int8mat(W,x,y); const original=Float32Array.from(y);
  codedmat(W,x,y); let max=0;
  for(let i=0;i<y.length;i++)max=Math.max(max,Math.abs(y[i]-original[i]));
  check.push(max);
}
for(const [name,fn] of [['int8',int8mat],['coded',codedmat],['coded_rot',codedrot]]){
  const samples=[];
  for(let t=0;t<5;t++){
    const a=performance.now(); for(let i=0;i<20;i++)sweep(fn);
    samples.push((performance.now()-a)/20);
  }
  const scaleBytes=matrices.reduce((n,m)=>n+m.W.scale.byteLength,0);
  console.log(JSON.stringify({name,ms_per_model_matvec_sweep:Math.min(...samples),resident_weight_bytes:
    (name==='int8'?matrices.reduce((n,m)=>n+m.W.q.byteLength,0):
    matrices.reduce((n,m)=>n+m.W.codes.byteLength,0)+lut.byteLength)+scaleBytes,
    max_abs_diff:name==='coded'?Math.max(...check):null}));
}
