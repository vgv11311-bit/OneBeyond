const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');
const path=require('node:path');
(async()=>{
 const jobId=process.argv[2],rawPath=process.argv[3];
 if(!/^[A-Za-z0-9_-]+$/.test(jobId||'')||!rawPath)throw new Error('Provide job ID and original PLY path');
 const base='http://127.0.0.1:5000';
 const job=await(await fetch(`${base}/api/reconstruct/status/${jobId}`)).json();
 const config=await(await fetch(base+job.gaussian.config_url)).json();
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const directory=path.resolve(__dirname,'../reports/gaussian-solid-oct19');await fs.mkdir(directory,{recursive:true});
 const report={jobId,cleanup:config.cleanup,alignment:config.alignment,errors:[],notice:'Matched raw/solid-completed viewer positions. Hidden wall and door are inferred semantic geometry, not a recovered photograph.'};
 try{
  for(const mode of ['raw','solid']){
   const context=await browser.newContext({viewport:{width:1280,height:800}});
   if(mode==='raw'){
    const raw=await fs.readFile(path.resolve(rawPath));
    await context.route('**'+job.gaussian.ply_url,route=>route.fulfill({body:raw,contentType:'application/octet-stream'}));
    await context.route('**'+job.gaussian.config_url,route=>route.fulfill({body:JSON.stringify({...config,solid:undefined,cleanup:undefined,notice:'Raw Gaussian reference; no solid completion.'}),contentType:'application/json'}));
   }
   const page=await context.newPage();page.on('pageerror',error=>report.errors.push(String(error)));
   page.on('console',message=>{if(message.type()==='error')report.errors.push(message.text().slice(0,2000));});
   await page.goto(base+'/spaces?renderer=three',{waitUntil:'domcontentloaded'});
   await page.getByTestId('space-'+jobId).click();
   const handle=await page.locator('iframe[title="Photographic room reconstruction"]').elementHandle();
   const frame=await handle.contentFrame();
   await frame.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
   const error=await frame.evaluate(()=>window.reviewError);if(error)throw new Error(error);
   await page.waitForTimeout(1200);
   if(mode==='solid'){
    report.defaultSurfaceMode=await frame.evaluate(()=>window.getReviewSurfaceMode());
    if(report.defaultSurfaceMode!=='solid')throw new Error('Geometry-first default missing');
    await page.screenshot({path:path.join(directory,'default-solid.png')});
    await frame.getByRole('button',{name:'Photographic detail',exact:true}).click();await page.waitForTimeout(500);
   }
   await page.screenshot({path:path.join(directory,mode+'-front.png')});
   if(mode==='solid'){
    await frame.getByRole('button',{name:'Solid surfaces',exact:true}).click();await page.waitForTimeout(500);
    await page.screenshot({path:path.join(directory,'solid-surfaces.png')});
    await frame.getByRole('button',{name:'Photographic detail',exact:true}).click();await page.waitForTimeout(500);
    report.solid=await frame.evaluate(()=>window.reviewSolid);
    if(report.solid?.opaqueEnvelopeSurfaces!==6||!report.solid?.inferredDoor||!report.solid?.backingHit)throw new Error('Solid enclosure missing');
    // Test actual exported collision helper on a controlled solid furniture box.
    report.collision=await frame.evaluate(async()=>{
     const THREE=await import('/api/gaussian-vendor/three/build/three.module.js');
     const {constrainMove}=await import('./support.js');
     const solid={bounds:{min:[-3,0,-5],max:[3,3,1]},obstacles:[{center:[0,.5,-2],size:[1,1,1],yaw_degrees:0}]};
     const from=new THREE.Vector3(0,1.2,-1.2);
     const furniture=constrainMove(from,new THREE.Vector3(0,1.2,-1.5),solid).toArray();
     const wall=constrainMove(new THREE.Vector3(2.8,1.2,0),new THREE.Vector3(5,1.2,3),solid).toArray();
     return {furnitureBlocked:Math.abs(furniture[2]-from.z)<1e-6,wallBounded:wall[0]<=2.84&&wall[2]<=.84};
    });
    if(!report.collision.furnitureBlocked||!report.collision.wallBounded)throw new Error('Collision proxy failed');
    report.solidSurfaceToggle=true;
   }
   await frame.evaluate(()=>window.turnReviewCamera(Math.PI));await page.waitForTimeout(1600);
   await page.screenshot({path:path.join(directory,mode+'-rear.png')});
   if(mode==='solid'){
    await frame.evaluate(bounds=>{window.setReviewCamera(document.querySelector('#view option').value);window.tryReviewMove([(bounds.min[0]+bounds.max[0])/2,1.2,bounds.max[2]*.7+bounds.min[2]*.3]);window.turnReviewCamera(Math.PI);},config.solid.bounds);
    await page.waitForTimeout(1600);await page.screenshot({path:path.join(directory,'solid-rear-wide.png')});
   }
   await frame.evaluate(()=>{window.setReviewCamera(document.querySelector('#view option').value);window.moveReviewCamera(.65);});await page.waitForTimeout(1600);
   await page.screenshot({path:path.join(directory,mode+'-offset.png')});
   await context.close();
  }
  if(report.errors.length)throw new Error(report.errors.join('; '));
 }catch(error){report.failure=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(directory,'browser-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
