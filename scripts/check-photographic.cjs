// Browser acceptance of the independent engine, upload flow and PLY-only viewer.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');
const path=require('node:path');
(async()=>{
 const jobId=process.argv[2];
 const intakeOnly=jobId==='--intake-only';
 if(!intakeOnly&&!/^[A-Za-z0-9_-]+$/.test(jobId||''))throw new Error('Provide a completed photographic QA job ID');
 const output=process.argv[3]||'photographic-independent-oct21-browser';
 if(!/^[\w-]+$/.test(output))throw new Error('Use a report directory name, not a path');
 const directory=path.resolve(__dirname,'../reports',output);
 await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900}});
 let inducingContextLoss=false;
 const errors=[],contextLossErrors=[];
 page.on('pageerror',e=>(inducingContextLoss?contextLossErrors:errors).push(String(e)));
 const report={jobId,errors,contextLossErrors,notice:'Browser integration and movement check. No unseen-room accuracy claim. Errors from intentionally losing the GPU context are reported separately.'};
 try{
  await page.goto('http://127.0.0.1:5173/studio',{waitUntil:'networkidle'});
  await page.getByRole('button',{name:/^Create 3D Space/}).click();
  await page.getByText('Overlapping photos → photographic 3D. No Groq or Gemini requests. Scale is unmeasured.',{exact:true}).waitFor();
  if(await page.getByText('Room Width — Measured or Estimated',{exact:true}).count())throw new Error('Photographic capture incorrectly requires measured semantic dimensions');
  await page.getByText('Photographic room',{exact:true}).waitFor();
  report.defaultPhotographic=true;
  await page.screenshot({path:path.join(directory,'upload-desktop.png')});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(directory,'upload-mobile.png')});
  report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  if(report.mobileOverflow)throw new Error('Mobile horizontal overflow');
  if(intakeOnly){report.intakeReady=true;return;}
  await page.setViewportSize({width:1365,height:900});
  await page.goto('http://127.0.0.1:5173/spaces',{waitUntil:'networkidle'});
  await page.getByTestId('space-'+jobId).click();
  const iframe=page.locator('iframe[title="Photographic room reconstruction"]');
  await iframe.waitFor();
  const frame=await (await iframe.elementHandle()).contentFrame();
  await frame.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  const failure=await frame.evaluate(()=>window.reviewError);
  if(failure)throw new Error(failure);
  await page.getByRole('link',{name:'Download splat',exact:true}).waitFor();
  if(await page.getByRole('button',{name:'Editable room',exact:true}).count())throw new Error('Standalone capture offers a nonexistent editable GLB');
  await page.waitForTimeout(1500);
  await page.screenshot({path:path.join(directory,'photographic.png')});
  const before=await frame.evaluate(()=>document.querySelector('canvas').toDataURL());
  const position=await frame.evaluate(()=>window.getReviewState().position);
  await frame.evaluate(()=>window.moveReviewCamera(.15));
  await page.waitForTimeout(700);
  const after=await frame.evaluate(()=>document.querySelector('canvas').toDataURL());
  report.movement=before!==after;
  report.cameraMoved=JSON.stringify(position)!==JSON.stringify(await frame.evaluate(()=>window.getReviewState().position));
  if(!report.movement||!report.cameraMoved)throw new Error('Camera navigation did not change the render');
  await page.screenshot({path:path.join(directory,'photographic-moved.png')});
  const keyboardStart=await frame.evaluate(()=>window.getReviewState().position);
  await frame.locator('canvas').focus();
  await page.keyboard.down('w');await page.waitForTimeout(1000);await page.keyboard.up('w');
  const keyboardEnd=await frame.evaluate(()=>window.getReviewState().position);
  report.keyboardMovement=JSON.stringify(keyboardStart)!==JSON.stringify(keyboardEnd);
  if(!report.keyboardMovement)throw new Error('WASD did not move the photographic camera');
  await page.screenshot({path:path.join(directory,'photographic-walk.png')});
  const capturedViews=await frame.locator('#view option').count();
  const merged=await frame.locator('#view').isHidden();
  const resetView=async()=>{if(merged)await frame.getByRole('button',{name:'Reset view',exact:true}).click();else await frame.locator('#view').selectOption({index:0});};
  if(!merged)await frame.locator('#view').selectOption({index:capturedViews-1});
  else {await resetView();report.singleMergedView=true;}
  await page.waitForTimeout(700);
  await page.screenshot({path:path.join(directory,'photographic-capture-last.png')});
  const depthWeight=await frame.evaluate(()=>window.getReviewState().surfaceWeight);
  const review=await frame.evaluate(()=>window.getReviewState());
  if(review.collision){
   if(depthWeight!==1)throw new Error('Walking still fades calibrated support to raw splats');
   const selected=await frame.locator('#view').inputValue();
   if(!merged&&review.surfaceAnchor!==selected)throw new Error('Wrong photo support at captured viewpoint');
   report.multiviewSupport=true;
   await resetView();
   const start=await frame.evaluate(()=>window.getReviewState().position);
   const result=await frame.evaluate(()=>window.tryReviewMove([0,0,-100]));
   report.forwardCollision={start,result};
   if(result[2]<review.collision.bounds.min[2])throw new Error('Escaped room boundary');
   if(result[2]>=start[2]-.1)throw new Error('Spawn is trapped by collision geometry');
   await page.waitForTimeout(500);
   await page.screenshot({path:path.join(directory,'photographic-furniture-stop.png')});
   const low=review.collision.bounds.min,high=review.collision.bounds.max,radius=review.collision.radius;
   for(const [axis,edge] of [[0,low[0]-20],[0,high[0]+20],[2,low[2]-20],[2,high[2]+20]]){
    await resetView();
    const target=[...start];target[axis]=edge;
    const p=await frame.evaluate(v=>window.tryReviewMove(v),target);
    if(p[axis]<low[axis]+radius-1e-5||p[axis]>high[axis]-radius+1e-5)throw new Error('Boundary tunnel');
   }
   report.closedBoundary=true;
   await resetView();
  }else if(depthWeight>.001)throw new Error('Single-anchor depth texture still covers another capture');
  report.anchorSurfaceSupport=true;
  report.captureSwitch=true;
  report.ready=true;
  inducingContextLoss=true;
  const lost=await frame.evaluate(()=>{const ext=document.querySelector('canvas').getContext('webgl2')?.getExtension('WEBGL_lose_context');if(!ext)return false;ext.loseContext();return true;});
  if(!lost)throw new Error('Could not exercise context-loss handling');
  await page.getByRole('button',{name:'Retry viewer',exact:true}).waitFor();
  if(await page.locator('canvas').count())throw new Error('Unexpected synthetic fallback for photographic-only capture');
  report.noInventedGlbFallback=true;
  if(errors.length)throw new Error(errors.join('; '));
 }catch(error){report.failure=String(error);process.exitCode=1;await page.screenshot({path:path.join(directory,'failure.png')}).catch(()=>{});}
 finally{await fs.writeFile(path.join(directory,intakeOnly?'intake-check.json':'browser-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
