const {chromium}=require('playwright'),fs=require('node:fs/promises');
(async()=>{
 const root='reports/hall-stage-audit-oct24/current-native',browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:960,height:704},deviceScaleFactor:1}),errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.route('**/viewer*',r=>r.fulfill({path:'scripts/native-ply-reference.html',contentType:'text/html'}));await page.route('**/config.json',r=>r.fulfill({path:root+'/matched-browser-config.json',contentType:'application/json'}));
 const ldr=process.argv.includes('--ldr'),prefix=ldr?'ldr-reference':'float-reference';
 const record={errors,poses:[],target:ldr?'RGBA8':'RGBA16F'};
 try{
  await page.goto('http://127.0.0.1:5000/api/spaces/bf37e7eb-f8f/gaussian/4fec4f3c7ac94b1a98c394210c5d6012/viewer'+(ldr?'?ldr=1':''));await page.waitForFunction(()=>window.referenceReady||window.referenceError,undefined,{timeout:120000});
  const error=await page.evaluate(()=>window.referenceError);if(error)throw Error(error);
  for(const name of ['00000.jpg','00002.jpg']){await page.evaluate(name=>window.setReferenceCamera(name),name);await page.screenshot({path:root+'/'+prefix+'-'+name+'.png',scale:'css'});record.poses.push(await page.evaluate(()=>window.referenceState()));}
 }catch(error){record.error=String(error);process.exitCode=1;}
 await fs.writeFile(root+'/'+prefix+'.json',JSON.stringify(record,null,2));await browser.close();console.log(JSON.stringify(record));
})();
