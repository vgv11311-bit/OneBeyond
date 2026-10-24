// External requests are blocked: prove saved-room replay needs neither cloud nor APIs.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const output=path.resolve(__dirname,'../reports/deadline-portable-check');await fs.mkdir(output,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const context=await browser.newContext({viewport:{width:1280,height:800}});
 const page=await context.newPage();const report={externalRequests:[],errors:[],freshReconstruction:false,qualityAccepted:false};
 await context.route('**/*',route=>{const url=new URL(route.request().url());if(url.origin!=='http://127.0.0.1:8768'){report.externalRequests.push(url.origin);return route.abort();}return route.continue();});
 page.on('pageerror',error=>report.errors.push(String(error)));
 try{
  await page.goto('http://127.0.0.1:8768');
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  const error=await page.evaluate(()=>window.reviewError);if(error)throw Error(error);
  report.initial=await page.evaluate(()=>window.getReviewState());
  await page.evaluate(()=>window.moveReviewCamera(.12));await page.waitForTimeout(300);
  report.translated=await page.evaluate(()=>window.getReviewState());
  if(JSON.stringify(report.initial.position)===JSON.stringify(report.translated.position))throw Error('Camera did not move');
  await page.getByRole('button',{name:'Geometry check',exact:true}).click();await page.waitForTimeout(250);
  if(!await page.evaluate(()=>window.getReviewState().geometryCheck))throw Error('Geometry check failed');
  await page.getByRole('button',{name:'Geometry check',exact:true}).click();await page.waitForTimeout(250);
  await page.screenshot({path:path.join(output,'saved-room-offline.png')});
  for(const route of ['/bundle.json','/../config.json.bak','/.env','/unknown/','/vendor/three/../../serve.py']){
   const response=await page.request.get('http://127.0.0.1:8768'+route);if(response.status()!==404)throw Error('Unexpected route exposed: '+route);
  }
  if(report.externalRequests.length||report.errors.length)throw Error('External request or page error');
  report.passed=true;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify({passed:report.passed,error:report.error,externalRequests:report.externalRequests}));
})();
