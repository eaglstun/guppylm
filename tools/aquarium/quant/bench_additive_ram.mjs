// Microbenchmark for direct two-code additive VQ matvec (8 values per chunk).
// Same matrix dimensions as the 8.87M-parameter aquarium model; synthetic codes.
import { performance } from 'node:perf_hooks';

const dims = [[4096,384], ...Array.from({length:6}, () => [[1152,384],[384,384],[768,384],[384,768]]).flat()];
const K=256, d=8;
const cats=[new Float32Array(K*d),new Float32Array(K*d)];
for(let s=0;s<2;s++)for(let i=0;i<K*d;i++)cats[s][i]=Math.sin(i*0.17+s*2)*0.15;

function direct(W,x,y){
  const {a,b,scale,rows,cols,tables}=W, nchunks=cols/d;
  for(let s=0;s<2;s++){
    const C=cats[s], T=tables[s];
    for(let c=0;c<nchunks;c++){
      const co=c*d, to=c*K;
      for(let k=0;k<K;k++){
        const p=k*d;
        T[to+k]=C[p]*x[co]+C[p+1]*x[co+1]+C[p+2]*x[co+2]+C[p+3]*x[co+3]
               +C[p+4]*x[co+4]+C[p+5]*x[co+5]+C[p+6]*x[co+6]+C[p+7]*x[co+7];
      }
    }
  }
  const A=tables[0],B=tables[1];
  for(let r=0,o=0;r<rows;r++,o+=nchunks){
    let sum=0;
    for(let c=0;c<nchunks;c++)sum+=A[c*K+a[o+c]]+B[c*K+b[o+c]];
    y[r]=sum*scale[r];
  }
}

function int8mat(W,x,y){
  const {q,scale,rows,cols}=W;
  for(let r=0,o=0;r<rows;r++,o+=cols){
    let a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for(let c=0;c<cols;c+=8){
      a0+=q[o+c]*x[c]; a1+=q[o+c+1]*x[c+1];
      a2+=q[o+c+2]*x[c+2]; a3+=q[o+c+3]*x[c+3];
      a4+=q[o+c+4]*x[c+4]; a5+=q[o+c+5]*x[c+5];
      a6+=q[o+c+6]*x[c+6]; a7+=q[o+c+7]*x[c+7];
    }
    y[r]=(a0+a1+a2+a3+a4+a5+a6+a7)*scale[r]/8;
  }
}

const matrices=dims.map(([rows,cols],i)=>{
  const chunks=rows*cols/d;
  const a=new Uint8Array(chunks),b=new Uint8Array(chunks),q=new Int8Array(rows*cols);
  for(let c=0;c<chunks;c++){
    a[c]=(c*37+i*11)&255; b[c]=(c*97+i*23)&255;
    for(let j=0;j<d;j++)q[c*d+j]=Math.round(8*(cats[0][a[c]*d+j]+cats[1][b[c]*d+j]));
  }
  const x=new Float32Array(cols);for(let c=0;c<cols;c++)x[c]=Math.sin(c*.013);
  const scale=new Float32Array(rows).fill(.1);
  const tables=[new Float32Array((cols/d)*K),new Float32Array((cols/d)*K)];
  return {W:{a,b,q,scale,rows,cols,tables},x,y:new Float32Array(rows)};
});
function sweep(fn){for(const {W,x,y} of matrices)fn(W,x,y);}
for(let i=0;i<5;i++){sweep(direct);sweep(int8mat);}
for(const [name,fn] of [['int8',int8mat],['additive',direct]]){
  const times=[];
  for(let rep=0;rep<5;rep++){
    const start=performance.now();for(let i=0;i<20;i++)sweep(fn);
    times.push((performance.now()-start)/20);
  }
  const scaleBytes=matrices.reduce((n,m)=>n+m.W.scale.byteLength,0);
  const matrixBytes=matrices.reduce((n,m)=>n+(name==='int8'?m.W.q.byteLength:m.W.a.byteLength+m.W.b.byteLength),0);
  const catalogBytes=name==='additive'?cats[0].byteLength+cats[1].byteLength:0;
  const reusableScratchBytes=name==='additive'?Math.max(...matrices.map(m=>m.W.tables[0].byteLength+m.W.tables[1].byteLength)):0;
  console.log(JSON.stringify({name,ms_per_model_matvec_sweep:Math.min(...times),resident_weight_bytes:matrixBytes+scaleBytes+catalogBytes,reusable_scratch_bytes:reusableScratchBytes}));
}
