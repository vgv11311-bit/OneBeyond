// Same-mesh geometry mode, reset/input and reload checks. Never reconstructs.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const job=process.argv[2],output=process.argv[3];
 if(!/^[\w-]+$/.test(job||'')||!/^[\w-]+$/.test(output||''))throw Error('Job and report name required');
 const directory=path.resolve(__dirname,'../reports',output);await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900}}),report={job,errors:[],qualityAccepted:false};
 page.on('pageerror',error=>report.errors.push(String(error)));
 const ready=async()=>{await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});const error=await page.evaluate(()=>window.reviewError);if(error)throw Error(error);};
 try{
  const response=await page.request.get('http://127.0.0.1:5000/api/reconstruct/status/'+job);if(!response.ok())throw Error('Saved job unavailable');
  const state=await response.json();report.revision=state.gaussian.revision;report.viewerUrl=state.gaussian.viewer_url;
  await page.goto('http://127.0.0.1:5000'+state.gaussian.viewer_url);await ready();
  report.initial=await page.evaluate(()=>window.getReviewState());
  await page.waitForTimeout(250);const idleBefore=await page.evaluate(()=>({state:window.getReviewState(),pixels:document.querySelector('canvas').toDataURL()}));
  await page.waitForTimeout(600);const idleAfter=await page.evaluate(()=>({state:window.getReviewState(),pixels:document.querySelector('canvas').toDataURL()}));
  report.idleDraws=idleAfter.state.drawnFrames-idleBefore.state.drawnFrames;
  report.idleSkippedFrames=idleAfter.state.idleSkippedFrames-idleBefore.state.idleSkippedFrames;
  if(report.idleDraws!==0||report.idleSkippedFrames<1||idleBefore.pixels!==idleAfter.pixels)throw Error('Idle mesh redraws or framebuffer changed');
  const photo=await page.evaluate(()=>document.querySelector('canvas').toDataURL());
  await page.getByRole('button',{name:'Geometry check',exact:true}).click();await page.waitForTimeout(350);
  report.geometry=await page.evaluate(()=>window.getReviewState());
  if(!report.geometry.geometryCheck||JSON.stringify(report.initial.position)!==JSON.stringify(report.geometry.position))throw Error('Geometry toggle changes camera or failed');
  if(photo===await page.evaluate(()=>document.querySelector('canvas').toDataURL()))throw Error('Geometry toggle did not change render');
  report.geometryPixels=await page.evaluate(()=>{
   const source=document.querySelector('canvas'),copy=document.createElement('canvas');copy.width=source.width;copy.height=source.height;
   const ctx=copy.getContext('2d');ctx.drawImage(source,0,0);const data=ctx.getImageData(0,0,copy.width,copy.height).data;
   let dark=0,colored=0;for(let i=0;i<data.length;i+=4){if(Math.max(data[i],data[i+1],data[i+2])<12)dark++;if(Math.max(data[i],data[i+1],data[i+2])-Math.min(data[i],data[i+1],data[i+2])>30)colored++;}
   return {darkFraction:dark/(data.length/4),coloredFraction:colored/(data.length/4)};
  });
  if(report.geometryPixels.darkFraction>.5||report.geometryPixels.coloredFraction<.1)throw Error('Untextured normals are missing or render black');
  await page.screenshot({path:path.join(directory,'untextured-front.png')});
  await page.evaluate(()=>window.moveReviewCamera(.12));await page.waitForTimeout(200);
  await page.screenshot({path:path.join(directory,'untextured-translated.png')});
  await page.getByRole('button',{name:'Geometry check',exact:true}).click();
  const name=await page.locator('#view').inputValue();
  await page.locator('canvas').focus();await page.keyboard.down('w');await page.waitForTimeout(150);
  await page.evaluate(name=>window.setReviewCamera(name),name);
  const reset=await page.evaluate(()=>window.getReviewState());await page.waitForTimeout(450);
  const still=await page.evaluate(()=>window.getReviewState());await page.keyboard.up('w');
  if(JSON.stringify(reset.position)!==JSON.stringify(still.position)||still.velocity.some(v=>Math.abs(v)>1e-10))throw Error('Held key drifts after reset');
  report.resetClearsHeldInput=true;
  await page.locator('canvas').focus();await page.keyboard.down('w');await page.waitForTimeout(100);
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));await page.keyboard.up('w');
  const blur=await page.evaluate(()=>window.getReviewState());await page.waitForTimeout(300);
  if(JSON.stringify(blur.position)!==JSON.stringify(await page.evaluate(()=>window.getReviewState().position)))throw Error('Movement persists after blur');
  report.blurStopsMovement=true;
  // Looking vertically and holding Space must not reduce horizontal movement
  // or make the walker fly through furniture.
  await page.evaluate(name=>window.setReviewCamera(name),name);const y=await page.evaluate(()=>window.getReviewState().position[1]);
  await page.locator('canvas').focus();await page.keyboard.down(' ');await page.waitForTimeout(150);await page.keyboard.up(' ');
  if(y!==await page.evaluate(()=>window.getReviewState().position[1]))throw Error('Walker can fly');
  await page.reload();await ready();report.reload=await page.evaluate(()=>window.getReviewState());
  if(report.reload.geometryCheck||!report.reload.enclosure)throw Error('Reload state/enclosure invalid');
  const draws=report.reload.drawnFrames;
  await page.setViewportSize({width:1000,height:700});await page.waitForTimeout(300);
  if(draws>=await page.evaluate(()=>window.getReviewState().drawnFrames))throw Error('Resize did not invalidate static render');
  report.resizeRedraw=true;
  if(report.errors.length)throw Error(report.errors.join(';'));
  report.passed=true;
 }catch(error){report.error=String(error);process.exitCode=1;await page.screenshot({path:path.join(directory,'failure.png')}).catch(()=>{});}
 finally{await fs.writeFile(path.join(directory,'viewer-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify({passed:report.passed,error:report.error,resetClearsHeldInput:report.resetClearsHeldInput,blurStopsMovement:report.blurStopsMovement},null,2));
})();
