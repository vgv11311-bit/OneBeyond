// Repeatable upward/rear transition and close-contact visual review.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');const path=require('node:path');
(async()=>{
 const job=process.argv[2],directory=path.resolve(__dirname,'../reports',process.argv[3]||'ceiling-contact-oct21');
 if(!/^[\w-]+$/.test(job||''))throw Error('Job required');
 await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1365,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 try{
  const state=await(await page.request.get('http://127.0.0.1:5000/api/reconstruct/status/'+job)).json();
  const revision=process.argv[4];if(revision&&!/^[a-f0-9]{32}$/.test(revision))throw Error('Invalid revision');
  if(revision){
   // Preview unpublished artifacts only inside this isolated browser. Preserve
   // production's current-revision access checks; never expose old captures.
   const prefix=state.gaussian.viewer_url.replace(/\/viewer$/,'');
   await page.route('**'+prefix+'/**',async route=>{
    const asset=new URL(route.request().url()).pathname.split('/').pop();
    if(asset==='config.json'||/^surface(?:-\d{2})?\.glb$/.test(asset)){
     const file=path.resolve(__dirname,'../backend/storage/spaces',job,'gaussian',revision,asset);
     await route.fulfill({body:await fs.readFile(file),contentType:asset.endsWith('.json')?'application/json':'model/gltf-binary'});
    }else await route.continue();
   });
  }
  await page.goto('http://127.0.0.1:5000'+state.gaussian.viewer_url);
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  const failure=await page.evaluate(()=>window.reviewError);if(failure)throw Error(failure);
  const name=await page.locator('#view').inputValue();
  const views=[];
  for(const [label,yaw,pitch] of [['front',0,0],['ceiling',0,1.1],['rear-transition',1.6,.4],['rear',Math.PI,0]]){
   await page.evaluate(({name,yaw,pitch})=>{window.setReviewCamera(name);window.turnReviewCamera(yaw);window.pitchReviewCamera(pitch);},{name,yaw,pitch});
   await page.waitForTimeout(500);
   await page.screenshot({path:path.join(directory,label+'.png')});
   views.push({label,state:await page.evaluate(()=>window.getReviewState())});
  }
  await page.evaluate(name=>window.setReviewCamera(name),name);
  const stop=await page.evaluate(()=>window.tryReviewMove([0,0,-100]));
  await page.waitForTimeout(500);await page.screenshot({path:path.join(directory,'contact.png')});
  const contact=await page.evaluate(()=>window.getReviewState());
  if(!contact.opaqueSurfaces||!contact.enclosure)throw Error('Close-up surface fading or missing enclosure');
  if(errors.length)throw Error(errors.join(';'));
  await fs.writeFile(path.join(directory,'seams-check.json'),JSON.stringify({job,views,stop,contact,errors},null,2));
 }finally{await browser.close();}
})();
