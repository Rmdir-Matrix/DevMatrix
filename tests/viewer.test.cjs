// Run with: node tests/viewer.test.cjs [real-export.matrix]
// Exercises the exact standalone HTML script using Node's browser-compatible
// Blob, gzip streams and WebCrypto; no server or third-party packages needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const html = fs.readFileSync('matrix-viewer.html','utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
function element(dataset={}) { return {dataset,value:'',hidden:false,className:'',textContent:'',files:[],classList:{add(){},remove(){}},listeners:{},addEventListener(name,fn){this.listeners[name]=fn},setAttribute(name,v){this[name]=v},click(){if(this.listeners.click)this.listeners.click()}}; }
const elements = Object.fromEntries(['drop','file','status','viewer','filter','output','section-hint','filename','filemeta','download'].map(x=>[x,element()]));
const tabs=['overview','samples','vocabulary','weights'].map(x=>element({tab:x}));
const document={getElementById:id=>elements[id],querySelectorAll:()=>tabs,createElement:()=>element()};
const context=vm.createContext({document,TextDecoder,DataView,Uint8Array,Array,Blob,DecompressionStream,atob,crypto:crypto.webcrypto,URL,setTimeout,console});
vm.runInContext(code,context);
function makeFile(bytes,name='demo.matrix') {let blob=new Blob([bytes]);blob.name=name;return blob;}
function synthetic() {
  const parts=[];
  const raw=x=>parts.push(Buffer.from(x));
  const u32=x=>{let b=Buffer.alloc(4);b.writeUInt32LE(x);parts.push(b)};
  const u16=x=>{let b=Buffer.alloc(2);b.writeUInt16LE(x);parts.push(b)};
  const meta=Buffer.from(JSON.stringify({name:'Test Matrix',core_replay:[{IDs:[2,7,8,3],LossStart:1}],recent_replay:[]}));
  raw('MXG3');u32(meta.length);raw(meta);
  [2,4,32,64,512].forEach(u32);raw(Buffer.alloc(8));u32(9);
  ['<pad>','<unk>','<bos>','<eos>','<user>','<assistant>','<self>','hello','world'].forEach((word,i)=>{let t=Buffer.from(word);u16(t.length);raw(t);u32(i)});
  for(let i=0;i<12+30*2;i++)u32(0);
  const payload=Buffer.concat(parts);
  const env={format:'matrix-share-v2',summary:{exported_name:'Test Matrix'},sha256_checksum:crypto.createHash('sha256').update(payload).digest('hex'),brain_payload:payload.toString('base64')};
  return {payload,env};
}
const gzip=env=>zlib.gzipSync(JSON.stringify(env));
(async()=>{
  const {env}=synthetic();
  const result=await context.parseMatrix(makeFile(gzip(env)));
  assert.equal(result.integrity,'SHA-256 verified');
  assert.equal(result.brain.vocab[7].word,'hello');
  assert.equal(result.brain.tensors.length,72);
  assert.match(context.makeSections(result,'demo.matrix').samples,/hello world/);
  await context.openFile(makeFile(gzip(env)));
  assert.equal(elements.viewer.hidden,false);
  assert.match(elements.output.textContent,/MODEL ARCHITECTURE/);
  tabs[1].click(); assert.match(elements.output.textContent,/hello world/);
  elements.filter.value='hello'; elements.filter.listeners.input(); assert.match(elements.output.textContent,/hello world/);
  let bad={...env,sha256_checksum:'0'.repeat(64)};
  await assert.rejects(context.parseMatrix(makeFile(gzip(bad))),/Checksum mismatch/);
  bad={...env,format:'future'}; await assert.rejects(context.parseMatrix(makeFile(gzip(bad))),/Unsupported file format/);
  await assert.rejects(context.parseMatrix(makeFile(Buffer.from('not gzip'))),/Could not open gzip/);
  let truncated=Buffer.from(env.brain_payload,'base64').subarray(0,80);
  bad={...env,brain_payload:truncated.toString('base64'),sha256_checksum:crypto.createHash('sha256').update(truncated).digest('hex')};
  await assert.rejects(context.parseMatrix(makeFile(gzip(bad))),/truncated|invalid/i);
  if(process.argv[2]) {
    const real=await context.parseMatrix(makeFile(fs.readFileSync(process.argv[2]),'real.matrix'));
    assert.equal(real.integrity,'SHA-256 verified');
    assert.ok(real.brain.layers>0 && real.brain.vocab.length>=7);
    assert.match(context.makeSections(real,'real.matrix').samples,/viewer test sample/);
  }
  console.log('Matrix viewer tests passed'+(process.argv[2]?' (including a real Go export)':''));
})().catch(e=>{console.error(e);process.exitCode=1});
