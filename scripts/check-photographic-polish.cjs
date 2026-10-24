const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');
const path=require('node:path');
(async()=>{
 const base='http://127.0.0.1:5000',id=process.argv[2]||'ef7f5704-f5b',label=process.argv[3]||'after';
 const job=await(await fetch(`${base}/api/reconstruct/status/${id}`)).json();
 const directory=path.resolve(__dirname,'../reports/photographic-polish-oct19');await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1280,height:800}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text().slice(0,1000));});
 const report={id,label,errors};
 try{
  await page.goto(base+job.gaussian.viewer_url,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  if(await page.evaluate(()=>window.reviewError))throw new Error(await page.evaluate(()=>window.reviewError));
  await page.evaluate(()=>window.setSurfaceMode('hybrid'));await page.waitForTimeout(2500);
  report.initialCamera=await page.locator('#view').inputValue();
  report.solid=await page.evaluate(()=>window.reviewSolid);
  for(const name of ['00000.jpg','00003.jpg']){
   await page.evaluate(name=>window.setReviewCamera(name),name);await page.waitForTimeout(1400);
   await page.screenshot({path:path.join(directory,`${label}-${name}.png`)});
  }
  await page.evaluate(()=>window.setReviewCamera('00000.jpg'));
  const before=await page.evaluate(()=>window.getReviewState?.());
  await page.mouse.move(650,450);await page.mouse.down();await page.mouse.move(850,510,{steps:12});await page.mouse.up();
  report.mouse=await page.evaluate(()=>window.getReviewState?.());
  await page.keyboard.press('Escape');await page.keyboard.down('KeyW');await page.waitForTimeout(650);await page.keyboard.up('KeyW');await page.waitForTimeout(400);
  report.walk={before,after:await page.evaluate(()=>window.getReviewState?.())};
  await page.screenshot({path:path.join(directory,`${label}-walk.png`)});
  await page.evaluate(()=>{window.setReviewCamera('00000.jpg');window.turnReviewCamera(Math.PI);});await page.waitForTimeout(1400);
  await page.screenshot({path:path.join(directory,`${label}-rear.png`)});
  report.rear=await page.evaluate(()=>window.getReviewState?.());
  report.overflow=await page.evaluate(()=>({x:document.documentElement.scrollWidth>innerWidth,y:document.documentElement.scrollHeight>innerHeight}));
  if(label!=='before'){
   if(report.walk.after.photoWeight!==1||report.rear.photoWeight!==1)throw new Error('Photographic mode faded while navigating');
   if(Math.abs(report.mouse.rotation[2])>1e-7)throw new Error('Mouse look introduced roll');
   if(report.overflow.x||report.overflow.y)throw new Error('Viewer overflow');
   await page.evaluate(()=>window.setReviewCamera('00000.jpg'));
   const start=await page.evaluate(()=>window.getReviewState().position);
   await page.locator('canvas').focus();await page.keyboard.down('KeyW');
   try{await page.waitForFunction(start=>Math.hypot(...window.getReviewState().position.map((v,i)=>v-start[i]))>.03,start,{timeout:10000});}finally{await page.keyboard.up('KeyW');}
   await page.waitForTimeout(250);
   report.keyboardDistance=await page.evaluate(start=>Math.hypot(...window.getReviewState().position.map((v,i)=>v-start[i])),start);
   if(report.keyboardDistance<.03)throw new Error('Keyboard walking did not move');
   await page.getByRole('button',{name:'Free look',exact:true}).click();
   await page.waitForFunction(()=>window.getReviewState().pointerLocked,undefined,{timeout:5000});
   report.pointerLock=true;await page.keyboard.press('Escape');
   await page.waitForFunction(()=>!window.getReviewState().pointerLocked,undefined,{timeout:5000});
   await page.setViewportSize({width:390,height:844});
   report.narrowOverflow=await page.evaluate(()=>({x:document.documentElement.scrollWidth>innerWidth,y:document.documentElement.scrollHeight>innerHeight}));
   if(report.narrowOverflow.x||report.narrowOverflow.y)throw new Error('Narrow viewport overflow');
   await page.setViewportSize({width:1280,height:800});
   await page.evaluate(()=>{window.setReviewCamera('00003.jpg');window.setSurfaceMode('hybrid');});await page.waitForTimeout(600);
   await page.screenshot({path:path.join(directory,`${label}-preview.png`)});
  }
  if(errors.length)throw new Error(errors.join('; '));
 }catch(error){report.failure=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(directory,`${label}-check.json`),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
