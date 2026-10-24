const {chromium}=require('playwright'),fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const out='reports/hall-resplat-audit-oct24',pointer=JSON.parse(await fs.readFile(out+'/candidate-pointer.json','utf8'));
 const browser=await chromium.launch({channel:'chrome',headless:true});const results=[];
 for(const [label,url] of [['before','/api/spaces/bf37e7eb-f8f/gaussian/4fec4f3c7ac94b1a98c394210c5d6012/viewer'],['after',pointer.viewer_url]]){
  const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:2}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('http://127.0.0.1:5000'+url);await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});await page.waitForFunction(()=>window.getReviewState?.().renderStats.triangles>0,undefined,{timeout:90000});
  const start=await page.evaluate(()=>window.getReviewState().position);const poses=[];
  for(const [name,yaw,offset] of [['entry',0,[0,0,0]],['right-rotation',-.75,[0,0,0]],['translate',-.75,[.6,0,-.6]],['return',0,[0,0,0]],['rear',Math.PI,[0,0,0]]]){
   await page.evaluate(({start,yaw,offset})=>{window.setReviewCamera('00000.jpg');window.turnReviewCamera(yaw);window.tryReviewMove(start.map((v,i)=>v+offset[i]));},{start,yaw,offset});await page.waitForTimeout(1000);
   poses.push({name,state:await page.evaluate(()=>window.getReviewState())});await page.screenshot({path:path.join(out,label+'-'+name+'.png'),scale:'css'});
  }
  const result={label,errors,poses};
  if(label==='after'){
   const toggle=page.getByRole('checkbox');if(await toggle.isChecked())throw Error('Inferred wall must default OFF');
   await toggle.check();await page.waitForTimeout(800);await page.screenshot({path:out+'/after-rear-optional-wall.png',scale:'css'});
   await page.getByRole('slider').focus();await page.getByRole('slider').press('ArrowRight');
   result.optionalWallVerified=true;await toggle.uncheck();
  }
  // Actual keyboard and pointer input, rather than only diagnostic setters.
  await page.evaluate(()=>window.setReviewCamera('00000.jpg'));const canvas=page.locator('canvas');await canvas.focus();await page.keyboard.down('KeyA');await page.waitForTimeout(250);await page.keyboard.up('KeyA');await page.waitForTimeout(300);
  result.keyboardEnd=await page.evaluate(()=>window.getReviewState());
  await page.mouse.move(600,400);await page.mouse.down();await page.mouse.move(750,420,{steps:8});await page.mouse.up();result.pointerEnd=await page.evaluate(()=>window.getReviewState());
  if(errors.length)throw Error(errors.join('\n'));results.push(result);await page.close();
 }
 await browser.close();await fs.writeFile(out+'/before-after.json',JSON.stringify(results,null,2));console.log(JSON.stringify(results.map(r=>({label:r.label,errors:r.errors,optionalWall:r.optionalWallVerified,keyboard:r.keyboardEnd.position,mouse:r.pointerEnd.rotation}))));
})();
