// Local CPU/COLMAP discovery and form checks; never submit or allocate a model.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const dataset=path.resolve(process.argv[2]);
 const directory=path.resolve(__dirname,'../reports/vggt-local-oct23-web');await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900},reducedMotion:'reduce'});
 const report={passed:false,writes:[],errors:[],modelRun:false,qualityAccepted:false};
 page.on('pageerror',error=>report.errors.push(String(error)));
 await page.route('**/api/**',route=>{
  if(route.request().method()!=='GET'){report.writes.push(route.request().url());return route.abort();}
  return route.continue();
 });
 try{
  const response=await page.request.get('http://127.0.0.1:5000/api/engines');
  report.engine=(await response.json()).engines.find(engine=>engine.id==='vggt_experimental');
  if(report.engine.device!=='cpu'||!report.engine.colmap_registration)throw Error('Local CPU/COLMAP configuration not exposed');
  await page.goto('http://127.0.0.1:5173/studio',{waitUntil:'networkidle'});
  await page.getByRole('button',{name:/^Create 3D Space/}).click();
  await page.getByRole('button',{name:/VGGT joint geometry/}).click();
  await page.getByText('COLMAP + VGGT joint camera/depth preview. Local CPU or CUDA; distortion is not verified fixed. No automatic cloud start.',{exact:true}).waitFor();
  const names=(await fs.readdir(dataset)).filter(name=>/\.jpg$/i.test(name)).sort().slice(0,5);
  if(names.length!==5)throw Error('Provide five saved photos');
  await page.locator('input[type=file]').setInputFiles(names.map(name=>path.join(dataset,name)));
  const submit=page.getByRole('button',{name:'Start 3D Reconstruction',exact:true});
  report.submitDisabled=await submit.isDisabled();
  if(report.submitDisabled===report.engine.can_run)throw Error('CPU readiness not reflected in form');
  if(!report.engine.can_run){
   // Available RAM changes between discovery and the app's own GET. Match the
   // stable status, not an earlier snapshot's rounded numeric memory amounts.
   await page.getByText(/^VGGT is installed locally\. Close unused apps:/).waitFor();
   if(report.engine.availability_code!=='vggt_local_memory_low')throw Error('Unexpected local configuration failure');
  }
  await page.getByRole('button',{name:'Recheck local memory',exact:true}).click();
  await page.getByRole('button',{name:'Recheck local memory',exact:true}).waitFor();
  report.recheckWithoutUpload=true;
  await page.screenshot({path:path.join(directory,'local-cpu-desktop.png')});
  await page.setViewportSize({width:390,height:844});
  report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  if(report.mobileOverflow)throw Error('Local readiness notice overflows mobile');
  await page.screenshot({path:path.join(directory,'local-cpu-mobile.png')});
  await page.getByRole('button',{name:/^Photographic room/}).click();
  report.defaultStillAvailable=await submit.isEnabled();
  if(!report.defaultStillAvailable||report.writes.length||report.errors.length)throw Error('Unexpected default regression or API write');
  report.passed=true;
 }catch(error){
  report.failure=String(error);throw error;
 }finally{
  await fs.writeFile(path.join(directory,'browser-check.json'),JSON.stringify(report,null,2));
  await browser.close();
 }
 console.log(JSON.stringify(report));
})().catch(error=>{console.error(error);process.exitCode=1;});
