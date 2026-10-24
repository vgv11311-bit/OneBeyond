// Browser route timing is diagnostic, not a claim of physical user-perceived FPS.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const job=process.argv[2],output=process.argv[3];
 if(!/^[\w-]+$/.test(job||'')||!/^[\w-]+$/.test(output||''))throw Error('Job and output name required');
 const directory=path.resolve(__dirname,'../reports',output);await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900},recordVideo:{dir:path.join(directory,'video'),size:{width:1365,height:900}}}),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>requests.push(r.url()));
 const report={job,errors,environment:'Headless Chrome on target laptop; requestAnimationFrame timing and page JS heap, not full process/GPU memory or physical input latency.'};
 try{
  const state=await(await page.request.get('http://127.0.0.1:5000/api/reconstruct/status/'+job)).json();
  let config=await(await page.request.get('http://127.0.0.1:5000'+state.gaussian.config_url)).json();
  const preview=process.env.ONE_BEYOND_PROFILE_REVISION;
  if(preview){
   if(!/^[a-f0-9]{32}$/.test(preview))throw Error('Invalid preview revision');
   const root=path.resolve(__dirname,'../backend/storage/spaces',job,'gaussian',preview),prefix=state.gaussian.viewer_url.replace(/\/viewer$/,'');
   config=JSON.parse(await fs.readFile(path.join(root,'config.json'),'utf8'));report.previewRevision=preview;
   await page.route('**'+prefix+'/**',async route=>{
    const asset=new URL(route.request().url()).pathname.split('/').pop();
    if(asset==='config.json'||/^surface(?:-\d{2})?\.glb$/.test(asset))await route.fulfill({body:await fs.readFile(path.join(root,asset)),contentType:asset.endsWith('.json')?'application/json':'model/gltf-binary'});
    else await route.continue();
   });
  }
  const baseline=process.argv[4];
  if(baseline){
   const source=path.resolve(baseline),prefix=state.gaussian.viewer_url.replace(/\/viewer$/,'');
   report.baselineSource=source;
   await page.route('**'+prefix+'/**',async route=>{
    const asset=new URL(route.request().url()).pathname.split('/').pop();
    const files={'viewer':'splat-review.html','support.js':'splat-solid.js','navigation.js':'splat-navigation.js','surface-confidence.js':'splat-surface-confidence.js'};
    if(files[asset]){
     let body=await fs.readFile(path.join(source,'scripts',files[asset]),'utf8');
     if(asset==='viewer')body=body.replaceAll('/vendor/','/api/gaussian-vendor/').replace("'/config.json'",JSON.stringify(prefix+'/config.json')).replace("'/scene.ply'",JSON.stringify(prefix+'/scene.ply'));
     await route.fulfill({body,contentType:asset==='viewer'?'text/html':'text/javascript'});
    }else await route.continue();
   });
  }
  const started=Date.now();await page.goto('http://127.0.0.1:5000'+state.gaussian.viewer_url);
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  if(await page.evaluate(()=>window.reviewError))throw Error(await page.evaluate(()=>window.reviewError));
  report.readyMs=Date.now()-started;const name=config.initial_camera,h=config.navigation.eye_height;
  await page.waitForTimeout(1000);
  report.route=[];
  for(const [label,offset,yaw,pitch] of [
   ['entrance',[0,0,0],0,0],['sofa-side',[-h*.35,0,-h*.25],-.55,0],['table-close',[0,0,-h*.6],0,-.35],
   ['turn-right',[h*.25,0,-h*.2],1.1,0],['rear',[0,0,0],Math.PI,0],['ceiling',[0,0,0],0,1.15],['floor',[0,0,0],0,-.9]]){
   await page.evaluate(({name,offset,yaw,pitch})=>{window.setReviewCamera(name);const p=window.getReviewState().position;window.tryReviewMove(p.map((v,i)=>v+offset[i]));window.turnReviewCamera(yaw);window.pitchReviewCamera(pitch);},{name,offset,yaw,pitch});
   const frames=await page.evaluate(()=>new Promise(resolve=>{const times=[];let last=performance.now(),start=last;function sample(now){times.push(now-last);last=now;if(now-start>=1200)resolve(times);else requestAnimationFrame(sample)}requestAnimationFrame(sample)}));
   frames.sort((a,b)=>a-b);report.route.push({label,medianFrameMs:frames[Math.floor(frames.length*.5)],p95FrameMs:frames[Math.floor(frames.length*.95)],frames:frames.length,state:await page.evaluate(()=>window.getReviewState())});
   await page.screenshot({path:path.join(directory,label+'.png')});
  }
  await page.evaluate(name=>window.setReviewCamera(name),name);await page.locator('canvas').focus();
  report.walkStart=await page.evaluate(()=>window.getReviewState());
  await page.keyboard.down('w');await page.waitForTimeout(2000);await page.keyboard.up('w');
  await page.waitForTimeout(350);report.walkEnd=await page.evaluate(()=>window.getReviewState());
  await page.mouse.move(700,450);await page.mouse.down();await page.mouse.move(1100,450,{steps:40});await page.mouse.up();
  await page.keyboard.down('d');await page.waitForTimeout(1500);await page.keyboard.up('d');
  await page.waitForTimeout(350);report.turnWalkEnd=await page.evaluate(()=>window.getReviewState());
  report.resources=await page.evaluate(()=>performance.getEntriesByType('resource').map(r=>({name:r.name,bytes:r.encodedBodySize,transferBytes:r.transferSize})).filter(r=>/\.(glb|ply|json)(?:\?|$)/.test(r.name)));
  const session=await page.context().newCDPSession(page);await session.send('Performance.enable');report.pageMetrics=(await session.send('Performance.getMetrics')).metrics.filter(m=>['JSHeapUsedSize','JSHeapTotalSize','Nodes','Documents'].includes(m.name));
  report.rawPlyRequested=requests.some(u=>/\/scene\.ply(?:\?|$)/.test(u));
  if(errors.length)throw Error(errors.join(';'));report.completed=true;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(directory,'profile.json'),JSON.stringify(report,null,2));const video=page.video();await page.close();if(video)await video.saveAs(path.join(directory,'walkthrough.webm'));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
