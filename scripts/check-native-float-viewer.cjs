// Local controlled renderer-only experiment. No training, no cloud activity.
const {chromium}=require('playwright'),fs=require('node:fs/promises');
(async()=>{
 const output='reports/quality-queue-oct24/native-float';await fs.mkdir(output,{recursive:true});
 const config=JSON.parse(await fs.readFile('reports/hall-stage-audit-oct24/current-native/matched-browser-config.json','utf8'));
 config.native_renderer='native-float-v1';
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:960,height:704},deviceScaleFactor:1}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.route('**/config.json',r=>r.fulfill({json:config}));
 await page.route('**/viewer',async r=>{const response=await r.fetch();await r.fulfill({response,body:(await response.text()).replace('renderer.setClearColor(0x171b20,1)','renderer.setClearColor(0,1)')});});
 const record={errors,poses:[]};
 try{
  await page.goto('http://127.0.0.1:5000/api/spaces/bf37e7eb-f8f/gaussian/4fec4f3c7ac94b1a98c394210c5d6012/viewer');
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:120000});
  if(await page.evaluate(()=>window.reviewError))throw Error(await page.evaluate(()=>window.reviewError));
  await page.waitForFunction(()=>window.getReviewState().nativeRenderer?.sorts>0,undefined,{timeout:30000});
  await page.addStyleTag({content:'header,.walk-hint{visibility:hidden!important}'});
  for(const camera of config.cameras){
   await page.evaluate(name=>window.setReviewCamera(name),camera.name);
   await page.waitForTimeout(1500);
   record.poses.push({name:camera.name,state:await page.evaluate(()=>window.getReviewState())});
   await page.screenshot({path:output+'/'+camera.name+'.png',scale:'css'});
  }
  await page.evaluate(()=>window.setReviewCamera('00000.jpg'));await page.waitForTimeout(500);
  await page.locator('canvas').focus();const before=await page.evaluate(()=>window.getReviewState());
  await page.keyboard.down('KeyW');await page.waitForTimeout(500);record.moving=await page.evaluate(()=>window.getReviewState());
  await page.screenshot({path:output+'/moving.png'});await page.keyboard.up('KeyW');await page.waitForTimeout(1800);
  record.settled=await page.evaluate(()=>window.getReviewState());await page.screenshot({path:output+'/settled.png'});
  await page.waitForTimeout(500);await page.screenshot({path:output+'/idle.png'});
  record.wMoved=before.position.some((v,i)=>Math.abs(v-record.moving.position[i])>1e-6);
  await page.evaluate(()=>window.turnReviewCamera(-.75));await page.waitForTimeout(1500);record.rotated=await page.evaluate(()=>window.getReviewState());await page.screenshot({path:output+'/rotated.png'});
  await page.evaluate(()=>window.setReviewCamera('00000.jpg'));await page.waitForTimeout(1500);await page.screenshot({path:output+'/returned.png'});
  record.buffer=await page.locator('canvas').evaluate(e=>({physical:[e.width,e.height],css:[e.clientWidth,e.clientHeight],dpr:devicePixelRatio}));
  if(!record.wMoved)throw Error('W did not move camera');if(errors.length)throw Error('Browser errors');
 }catch(e){record.error=String(e);process.exitCode=1;}
 await fs.writeFile(output+'/result.json',JSON.stringify(record,null,2));await browser.close();console.log(JSON.stringify(record));
})();
