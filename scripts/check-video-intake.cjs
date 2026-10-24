// Real browser upload + real FFmpeg metadata validation; intentionally no
// duplicate heavy training. Fixture is a slideshow, not a capture benchmark.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');const path=require('node:path');
(async()=>{
 const directory=path.resolve(__dirname,'../reports/video-intake-oct21');
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900}});
 const report={notice:'Encoded five-photo slideshow tests video intake only, not genuine walkthrough reconstruction quality.'};
 try{
  await page.goto('http://127.0.0.1:5000/studio',{waitUntil:'networkidle'});
  await page.getByRole('button',{name:/^Create 3D Space/}).click();
  await page.getByRole('button',{name:'Walkthrough video',exact:true}).click();
  const input=page.locator('input[type=file]');
  if((await input.getAttribute('accept')).includes('.jpg'))throw new Error('Video picker still requests photos');
  if(await input.getAttribute('multiple')!==null)throw new Error('Video mode accepts multiple clips');
  await input.setInputFiles(path.join(directory,'five-view-fixture.mp4'));
  await page.getByRole('button',{name:'Walkthrough video',exact:true}).waitFor();
  await page.screenshot({path:path.join(directory,'video-selected.png')});
  let request;
  await page.route('**/api/reconstruct',async route=>{
   request=route.request().postDataJSON();
   await route.fulfill({status:400,contentType:'application/json',body:JSON.stringify({error:'Video intake QA complete; duplicate training intentionally not started.'})});
  });
  const uploaded=page.waitForResponse(r=>r.url().endsWith('/api/upload')&&r.request().method()==='POST');
  await page.locator('button[type=submit]').click();
  const response=await uploaded;const job=await response.json();
  if(response.status()!==201||job.input_type!=='video')throw new Error(JSON.stringify(job));
  await page.waitForTimeout(500);
  if(request?.engine_id!=='photographic'||request?.job_id!==job.job_id)throw new Error('Video did not route to photographic reconstruction');
  report.upload=job;report.request=request;report.passed=true;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(directory,'browser-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
