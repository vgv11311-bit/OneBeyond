// Exercise real browser input separately from synthetic navigation APIs.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const job=process.argv[2],output=process.argv[3];
 if(!/^[\w-]+$/.test(job||'')||!/^[\w-]+$/.test(output||''))throw Error('Job and output required');
 const directory=path.resolve(__dirname,'../reports',output);await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true}),page=await browser.newPage({viewport:{width:1100,height:800}});
 const report={errors:[],checks:[]};page.on('pageerror',e=>report.errors.push(String(e)));
 const assert=(condition,label)=>{if(!condition)throw Error(label);report.checks.push(label);};
 try{
  const status=await(await page.request.get('http://127.0.0.1:5000/api/reconstruct/status/'+job)).json();
  await page.goto('http://127.0.0.1:5000'+status.gaussian.viewer_url);
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  if(await page.evaluate(()=>window.reviewError))throw Error(await page.evaluate(()=>window.reviewError));
  const state=()=>page.evaluate(()=>window.getReviewState()),reset=()=>page.locator('#reset').click();
  const initial=await state();
  await page.mouse.move(650,400);await page.mouse.down();await page.mouse.move(750,430,{steps:10});await page.mouse.up();
  const drag=await state();assert(Math.abs(drag.rotation[1]-initial.rotation[1])>.1,'drag changes heading');
  await page.mouse.move(850,500);await page.waitForTimeout(150);
  const released=await state();assert(Math.abs(released.rotation[1]-drag.rotation[1])<1e-8,'released pointer does not keep looking');
  await reset();const again=await state();
  assert(again.position.every((v,i)=>Math.abs(v-initial.position[i])<1e-8),'reset restores capture position');
  assert(again.rotation.every((v,i)=>Math.abs(v-initial.rotation[i])<1e-8),'reset restores calibrated orientation');
  await page.locator('canvas').focus();await page.keyboard.down('ArrowUp');await page.waitForTimeout(400);
  await page.evaluate(()=>dispatchEvent(new Event('blur')));const blurred=await state();
  await page.waitForTimeout(300);await page.keyboard.up('ArrowUp');const stopped=await state();
  assert(stopped.position.every((v,i)=>Math.abs(v-blurred.position[i])<1e-8),'blur clears movement and momentum');
  assert(stopped.velocity.every(v=>Math.abs(v)<1e-8),'blur clears velocity');
  // Chromium allows pointer lock after an actual button click in this harness.
  await page.locator('#lock').click();await page.waitForTimeout(150);
  const locked=await state();assert(locked.pointerLocked,'pointer lock enters');
  await page.evaluate(()=>document.exitPointerLock());await page.waitForTimeout(150);
  assert(!(await state()).pointerLocked,'pointer lock exits');
  await page.locator('#lock').click();await page.waitForTimeout(150);
  assert((await state()).pointerLocked,'pointer lock reenters');
  await page.keyboard.press('Escape');await page.evaluate(()=>document.exitPointerLock());await page.waitForTimeout(100);
  await reset();await page.screenshot({path:path.join(directory,'controls.png')});
  assert(!report.errors.length,'no browser exceptions');report.passed=true;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(directory,'controls-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
