// Read-only browser/API integration: never submits a reconstruction or starts GPU.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const directory=path.resolve(__dirname,'../reports/vggt-web-oct22/browser');await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900},reducedMotion:'reduce'});
 const report={errors:[],writes:[],gpuUsed:false,qualityAccepted:false};
 page.on('pageerror',error=>report.errors.push(String(error)));
 await page.route('**/api/**',route=>{if(route.request().method()!=='GET'){report.writes.push(route.request().url());return route.abort();}return route.continue();});
 try{
  const response=await page.request.get('http://127.0.0.1:5000/api/engines');
  const engines=(await response.json()).engines;
  report.engine=engines.find(engine=>engine.id==='vggt_experimental');
  if(!report.engine||report.engine.can_run||report.engine.is_default)throw Error('VGGT must be discoverable but inactive on the laptop');
  await page.goto('http://127.0.0.1:5173/studio',{waitUntil:'networkidle'});
  await page.getByRole('button',{name:/^Create 3D Space/}).click();
  const card=page.getByRole('button',{name:/VGGT joint geometry/});await card.click();
  await page.getByText(/Integrated but inactive/).waitFor();
  if(await page.getByText('Room Width — Measured or Estimated',{exact:true}).count())throw Error('VGGT incorrectly asks for measured semantic dimensions');
  if(await card.getAttribute('aria-pressed')!=='true')throw Error('Experimental engine was not selected');
  const dataset=path.resolve(process.argv[2]);
  const files=(await fs.readdir(dataset)).filter(name=>/\.jpg$/i.test(name)).sort().slice(0,5).map(name=>path.join(dataset,name));
  if(files.length!==5)throw Error('Provide five existing photos for local form validation');
  await page.locator('input[type=file]').setInputFiles(files);
  const submit=page.getByRole('button',{name:'Start 3D Reconstruction',exact:true});
  if(!await submit.isDisabled())throw Error('Unavailable CUDA engine can submit');
  await page.screenshot({path:path.join(directory,'experimental-desktop.png')});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(directory,'experimental-mobile.png')});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
  await page.getByRole('button',{name:/^Photographic room/}).click();
  if(!await submit.isEnabled())throw Error('Default photographic mode was disabled by VGGT gating');
  await page.getByRole('button',{name:'Cancel',exact:true}).click();
  await page.getByRole('button',{name:/^Create 3D Space/}).click();
  if(await page.getByRole('button',{name:/^Photographic room/}).getAttribute('aria-pressed')!=='true')throw Error('Reopen did not reset to original default');
  if(report.writes.length||report.errors.length)throw Error('Unexpected API write or browser error');
  report.passed=true;report.unavailableSubmitBlocked=true;report.defaultPreserved=true;
 }catch(error){report.error=String(error);process.exitCode=1;await page.screenshot({path:path.join(directory,'failure.png')}).catch(()=>{});}
 finally{await fs.writeFile(path.join(directory,'report.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify({passed:report.passed,error:report.error,writes:report.writes}));
})();
