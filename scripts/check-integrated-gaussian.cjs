// Browser acceptance of the actual React -> Flask -> Spark route and GLB switch.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');
const path=require('node:path');
(async()=>{
 const jobId=process.argv[2];if(!/^[A-Za-z0-9_-]+$/.test(jobId||''))throw new Error('Provide a job ID');
 const unity=process.argv[3]==='unity';
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1280,height:800}});
 const errors=[];page.on('pageerror',error=>errors.push(String(error)));
 const report={jobId,editableRenderer:unity?'Unity':'Three',errors,notice:'Saved semantic GLB replay plus freshly trained Gaussian companion; browser integration check, not fresh Groq analysis or unseen-room accuracy.'};
 const directory=path.resolve(__dirname,'../reports/gaussian-integration-oct19'+(unity?'-unity':''));await fs.mkdir(directory,{recursive:true});
 await page.addInitScript(()=>{window.unityEvents=[];document.addEventListener('one-beyond-unity',event=>{try{window.unityEvents.push(JSON.parse(event.detail));}catch{};},true);});
 try{
  await page.goto('http://127.0.0.1:5000/spaces'+(unity?'':'?renderer=three'),{waitUntil:'domcontentloaded'});
  await page.getByTestId('space-'+jobId).click();
  const iframe=page.locator('iframe[title="Photographic room reconstruction"]');
  await iframe.waitFor();
  let frame=await (await iframe.elementHandle()).contentFrame();
  await frame.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  if(await frame.evaluate(()=>window.reviewError))throw new Error('Splat failed');
  await frame.evaluate(()=>window.setSurfaceMode?.('hybrid'));
  await page.waitForTimeout(2000);
  await page.screenshot({path:path.join(directory,'photographic.png')});
  const before=await frame.evaluate(()=>document.querySelector('canvas').toDataURL());
  await frame.evaluate(()=>window.moveReviewCamera(.25));await page.waitForTimeout(700);
  const after=await frame.evaluate(()=>document.querySelector('canvas').toDataURL());
  if(before===after)throw new Error('Splat movement did not change canvas');
  report.splatMovement=true;
  await page.getByRole('button',{name:'Editable room',exact:true}).click();
  await page.locator('canvas').waitFor({timeout:90000});
  if(unity)await page.waitForFunction(()=>window.unityEvents.some(event=>event.type==='loaded'),undefined,{timeout:120000});
  await page.waitForTimeout(5000);
  await page.screenshot({path:path.join(directory,'editable.png')});
  report.editableSwitch=true;
  await page.getByRole('button',{name:'Photographic room',exact:true}).click();
  await page.locator('iframe[title="Photographic room reconstruction"]').waitFor();
  frame=await (await page.locator('iframe').elementHandle()).contentFrame();
  await frame.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  report.returnSwitch=true;
  // Force an actual WebGL context-loss event, not a fabricated success signal.
  const lost=await frame.evaluate(()=>{const gl=document.querySelector('canvas').getContext('webgl2');const extension=gl?.getExtension('WEBGL_lose_context');if(!extension)return false;extension.loseContext();return true;});
  if(!lost)throw new Error('Context-loss extension unavailable; fallback check not exercised');
  await page.getByText('Photographic rendering is unavailable on this device.',{exact:true}).waitFor();
  await page.locator('canvas').waitFor({timeout:90000});
  if(unity)await page.waitForFunction(()=>window.unityEvents.filter(event=>event.type==='loaded').length>=2,undefined,{timeout:120000});
  report.forcedContextLossFallback=true;
  if(errors.length)throw new Error('Browser errors: '+errors.join('; '));
 }catch(error){report.failure=String(error);process.exitCode=1;await page.screenshot({path:path.join(directory,'failure.png')}).catch(()=>{});}
 finally{await fs.writeFile(path.join(directory,'browser-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
