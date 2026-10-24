// Read-only candidate comparison. No API writes, uploads or saved-room mutation.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const job=process.argv[2],candidate=path.resolve(process.argv[3]);
 if(!/^[\w-]+$/.test(job||''))throw Error('Provide saved job ID and candidate folder');
 const config=JSON.parse(await fs.readFile(path.join(candidate,'config.json'),'utf8'));
 const status=await(await fetch(`http://127.0.0.1:5000/api/reconstruct/status/${job}`)).json();
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const report={writes:[],runs:[],qualityAccepted:false,savedRoomChanged:false};
 try{
  for(const label of ['baseline','candidate']){
   const page=await browser.newPage({viewport:{width:1280,height:800}}),errors=[];
   page.on('pageerror',error=>errors.push(String(error)));
   await page.route('**/api/**',route=>{
    const request=route.request();
    if(request.method()!=='GET'){report.writes.push(request.url());return route.abort();}
    if(label==='candidate'&&request.url().endsWith('/config.json'))return route.fulfill({path:path.join(candidate,'config.json'),contentType:'application/json'});
    if(label==='candidate'&&request.url().endsWith('/surface-99.glb'))return route.fulfill({path:path.join(candidate,'surface-99.glb'),contentType:'model/gltf-binary'});
    return route.continue();
   });
   await page.goto('http://127.0.0.1:5000'+status.gaussian.viewer_url,{waitUntil:'domcontentloaded'});
   await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
   const error=await page.evaluate(()=>window.reviewError);if(error)throw Error(error);
   const states=[];
   async function capture(name){
    await page.waitForTimeout(500);
    const state=await page.evaluate(()=>window.getReviewState());states.push({name,...state});
    if(state.photoWeight!==1||!state.opaqueSurfaces)throw Error('Candidate became translucent');
    await page.screenshot({path:path.join(candidate,`${label}-${name}.png`)});
   }
   await page.evaluate(name=>window.setReviewCamera(name),config.cameras[0].name);await capture('anchor');
   await page.evaluate(distance=>window.moveReviewCamera(distance),config.navigation.eye_height*.3);await capture('side');
   await page.locator('canvas').focus();await page.keyboard.down('w');await page.waitForTimeout(900);await page.keyboard.up('w');await capture('approach');
   await page.evaluate(()=>window.turnReviewCamera(Math.PI));await capture('rear');
   await page.evaluate(name=>window.setReviewCamera(name),config.cameras[0].name);
   await page.mouse.move(900,500);await page.mouse.down();await page.mouse.move(900,270,{steps:8});await page.mouse.up();await capture('ceiling');
   if(errors.length)throw Error(errors.join('\n'));
   report.runs.push({label,states,errors});await page.close();
  }
  if(report.writes.length)throw Error('Unexpected API write');
 }finally{
  await fs.writeFile(path.join(candidate,'browser-check.json'),JSON.stringify(report,null,2));
  await browser.close();
 }
 console.log(JSON.stringify({runs:report.runs.length,writes:report.writes.length,qualityAccepted:false}));
})().catch(error=>{console.error(error);process.exitCode=1;});
