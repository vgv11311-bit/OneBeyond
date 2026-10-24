// Form/discovery smoke test only. Never submits uploads or starts a GPU.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const out=path.resolve(__dirname,'../reports/resplat-oct23');await fs.mkdir(out,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900},reducedMotion:'reduce'});
 const report={passed:false,writes:[],errors:[],inferenceRan:false,qualityAccepted:false};
 page.on('pageerror',e=>report.errors.push(String(e)));
 await page.route('**/api/**',route=>{if(route.request().method()!=='GET'){report.writes.push(route.request().url());return route.abort();}return route.continue();});
 try{
  const engines=await (await page.request.get('http://127.0.0.1:5000/api/engines')).json();
  report.engine=engines.engines.find(e=>e.id==='resplat_experimental');
  if(!report.engine||report.engine.is_default||report.engine.can_run)throw Error('Expected explicit experiment, unavailable laptop CUDA');
  await page.goto('http://127.0.0.1:5000/studio');
  await page.getByRole('button',{name:/^Create 3D Space/}).first().click();
  await page.getByRole('button',{name:/^COLMAP \+ ReSplat/}).click();
  await page.getByText(/^ReSplat is downloaded\./).waitFor();
  const input=path.resolve(process.argv[2]);
  const names=(await fs.readdir(input)).filter(n=>/\.jpg$/i.test(n)).sort().slice(0,5);
  if(names.length!==5)throw Error('Expected five real saved photos');
  await page.locator('input[type=file]').setInputFiles(names.map(n=>path.join(input,n)));
  const submit=page.getByRole('button',{name:'Start 3D Reconstruction',exact:true});
  if(!await submit.isDisabled())throw Error('CUDA unavailable but submit enabled');
  if(await page.getByText('Room Width — Measured or Estimated',{exact:true}).count())throw Error('Unmeasured experimental scale mislabelled');
  await page.getByText(/^ReSplat is downloaded\./).scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(out,'web-desktop.png')});
  await page.setViewportSize({width:390,height:844});
  await page.getByText(/^ReSplat is downloaded\./).scrollIntoViewIfNeeded();
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
  await page.screenshot({path:path.join(out,'web-mobile.png')});
  await page.getByRole('button',{name:/^Photographic room/}).click();
  if(!await submit.isEnabled())throw Error('Existing default no longer usable');
  if(report.writes.length||report.errors.length)throw Error('Unexpected write or page error');
  report.passed=true;
 }catch(e){report.failure=String(e);process.exitCode=1;}
 finally{await fs.writeFile(path.join(out,'browser-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report));
})();
