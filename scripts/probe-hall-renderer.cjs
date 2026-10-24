// Controlled, read-only browser variants. All use the SAME PLY, camera and path.
const {chromium}=require('playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const out=path.resolve('reports/hall-resplat-audit-oct24');
 const url='http://127.0.0.1:5000/api/spaces/bf37e7eb-f8f/gaussian/4fec4f3c7ac94b1a98c394210c5d6012/viewer';
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const records=[];
 for(const variant of ['baseline','native-dpr','aa-match','depth-sort','extended']){
  const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:2});const requests=[],errors=[];
  page.on('request',r=>requests.push(r.url()));page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/viewer',async route=>{
   const response=await route.fetch();let html=await response.text();
   if(variant==='native-dpr')html=html.replace('Math.min(devicePixelRatio,1.5)','Math.min(devicePixelRatio,2)');
   if(variant==='aa-match')html=html.replace('depthTest:!!(config.native_splat_display&&config.navigation)','blurAmount:.1,depthTest:!!(config.native_splat_display&&config.navigation)');
   if(variant==='depth-sort')html=html.replace('depthTest:!!(config.native_splat_display&&config.navigation)','sortRadial:false,depthTest:!!(config.native_splat_display&&config.navigation)');
   if(variant==='extended')html=html.replace('depthTest:!!(config.native_splat_display&&config.navigation)','accumExtSplats:true,depthTest:!!(config.native_splat_display&&config.navigation)').replace("new SplatMesh({url:'/scene.ply'})","new SplatMesh({url:'/scene.ply',extSplats:true})");
   html=html.replace('window.reviewReady=true;',`window.hallProbe=()=>({css:[innerWidth,innerHeight],buffer:[canvas.width,canvas.height],dpr:devicePixelRatio,pixelRatio:renderer.getPixelRatio(),projection:camera.projectionMatrix.toArray(),spark:{blur:spark?.blurAmount,preBlur:spark?.preBlurAmount,radial:spark?.sortRadial,lod:spark?.enableLod,active:spark?.activeSplats,extended:spark?.accumExtSplats,aperture:spark?.apertureAngle},photoMeshes:hasPhotoMesh});window.reviewReady=true;`);
   await route.fulfill({response,body:html});
  });
  const record={variant,errors,poses:[]};
  try{
   await page.goto(url);await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
   await page.waitForFunction(()=>window.getReviewState?.().renderStats.triangles>0,undefined,{timeout:90000});
   const start=await page.evaluate(()=>window.getReviewState().position);
   const poses=[['entry',0,[0,0,0]],['right-rotation',-.75,[0,0,0]],['translate',-.75,[.6,0,-.6]],['return',0,[0,0,0]],['rear',Math.PI,[0,0,0]]];
   for(const [name,yaw,offset] of poses){
    await page.evaluate(({yaw,offset,start})=>{window.setReviewCamera('00000.jpg');window.turnReviewCamera(yaw);window.tryReviewMove(start.map((v,i)=>v+offset[i]));},{yaw,offset,start});
    await page.waitForTimeout(1200);
    record.poses.push({name,state:await page.evaluate(()=>window.getReviewState()),renderer:await page.evaluate(()=>window.hallProbe())});
    await page.screenshot({path:path.join(out,variant+'-'+name+'.png'),scale:'css'});
   }
   record.plyRequested=requests.some(r=>r.endsWith('/scene.ply'));record.meshRequested=requests.some(r=>r.endsWith('.glb'));
  }catch(e){record.error=String(e);}
  records.push(record);await page.close();
 }
 await fs.writeFile(path.join(out,'renderer-probes.json'),JSON.stringify(records,null,2));await browser.close();
 console.log(JSON.stringify(records.map(r=>({variant:r.variant,error:r.error,errors:r.errors,renderer:r.poses[0]?.renderer,ply:r.plyRequested,mesh:r.meshRequested}))));
})();
