// Isolated local splat renderer check; does not change the production app.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');
const path=require('node:path');
const crypto=require('node:crypto');
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:768,height:576}});
 const errors=[];page.on('pageerror',error=>errors.push(String(error)));
 const directory=path.resolve(__dirname,'../reports/local-quality-oct19');await fs.mkdir(directory,{recursive:true});
 const result={errors,notice:'Local saved-splat rendering and movement check, not production integration or full coverage proof.'};
 try{
  await page.goto('http://127.0.0.1:8766/',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  const failure=await page.evaluate(()=>window.reviewError);if(failure)throw new Error(failure);
  await page.locator('header').evaluate(node=>node.style.display='none');
  const cameras=await page.locator('#view option').evaluateAll(options=>options.map(option=>option.value));
  result.cameras=[];
  for(const camera of cameras){
   await page.evaluate(name=>window.setReviewCamera(name),camera);await page.waitForTimeout(2500);
   const filename=`splat-browser-${camera.replace(/\W/g,'_')}.png`;
   await page.screenshot({path:path.join(directory,filename)});result.cameras.push({camera,screenshot:filename});
  }
  await page.evaluate(()=>window.setReviewCamera(document.querySelector('#view option').value));await page.waitForTimeout(2000);
  const before=await page.screenshot();await page.evaluate(()=>window.moveReviewCamera(.6));await page.waitForTimeout(2500);
  const after=await page.screenshot({path:path.join(directory,'splat-browser-offset.png')});
  result.movementChangedView=crypto.createHash('sha256').update(before).digest('hex')!==crypto.createHash('sha256').update(after).digest('hex');
  if(!result.movementChangedView)throw new Error('Camera movement did not change the view');
  await page.evaluate(()=>window.moveReviewCamera(2.5));await page.waitForTimeout(2500);
  await page.screenshot({path:path.join(directory,'splat-browser-offset-large.png')});
  result.largeOffsetSfmUnits=3.1; // Not metres; no measured scale was supplied.
 }catch(error){result.failure=String(error);await page.screenshot({path:path.join(directory,'splat-browser-failure.png')}).catch(()=>{});process.exitCode=1}
 finally{await fs.writeFile(path.join(directory,'browser-check.json'),JSON.stringify(result,null,2));await browser.close()}
 console.log(JSON.stringify(result,null,2));
})();
