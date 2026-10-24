// Own-run archived native comparison; it is NOT the current scene's native render.
const {chromium}=require('playwright'),fs=require('node:fs/promises');
(async()=>{
 const root='reports/hall-stage-audit-oct24',current=process.argv.includes('--current'),control=root+(current?'/current-native/browser':'/archived-control');await fs.mkdir(control,{recursive:true});
 const configPath=current?root+'/current-native/matched-browser-config.json':control+'/config.json';
 const fixedPath=current?root+'/baseline/bf37e7eb-f8f/scene.ply':control+'/scene-fixed.ply';
 const rawPath=current?'backend/storage/spaces/bf37e7eb-f8f/resplat/f10a963a2724491b9d2e3a10b6be0394/result/gaussians.ply':control+'/scene-raw.ply';
 const browser=await chromium.launch({channel:'chrome',headless:true}),report=[];
 const viewer='http://127.0.0.1:5000/api/spaces/bf37e7eb-f8f/gaussian/4fec4f3c7ac94b1a98c394210c5d6012/viewer';
 for(const variant of ['raw','fixed','fixed-aa','fixed-aa-depth','fixed-aa-depth-support','extended-source','extended-source-accum']){
  const page=await browser.newPage({viewport:{width:960,height:704},deviceScaleFactor:1}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/config.json',route=>route.fulfill({path:configPath,contentType:'application/json'}));
  await page.route('**/scene.ply',route=>route.fulfill({path:variant==='raw'?rawPath:fixedPath,contentType:'application/octet-stream'}));
  await page.route('**/viewer',async route=>{
   const res=await route.fetch();let html=await res.text();
   const extra={};if(variant.includes('aa'))extra.blurAmount=.1;if(variant.includes('depth'))extra.sortRadial=false;if(variant.includes('support'))extra.maxStdDev=3;
   html=html.replace('depthTest:!!(config.native_splat_display&&config.navigation)',Object.entries(extra).map(([k,v])=>k+':'+v+',').join('')+'depthTest:false');
   if(variant.startsWith('extended')){let count=0;html=html.replace(/new SplatMesh\(\{url:('[^']*'|"[^"]*")\}\)/,(_,url)=>{count++;return `new SplatMesh({url:${url},extSplats:true})`;});if(count!==1)throw Error('Extended model replacement failed');}
   if(variant.includes('accum'))html=html.replace('depthTest:false','accumExtSplats:true,depthTest:false');
   html=html.replace('renderer.setClearColor(0x171b20,1)','renderer.setClearColor(0,1)');
   html=html.replace('window.reviewReady=true;',`window.auditRenderer=()=>({buffer:[canvas.width,canvas.height],css:[innerWidth,innerHeight],projection:camera.projectionMatrix.toArray(),pixelRatio:renderer.getPixelRatio(),spark:{blur:spark.blurAmount,radial:spark.sortRadial,maxStdDev:spark.maxStdDev,active:spark.activeSplats,source:model.splats.constructor.name,accumExt:spark.accumExtSplats},native:config.native_splat_display,mesh:hasPhotoMesh});window.reviewReady=true;`);
   await route.fulfill({response:res,body:html});
  });
  const record={variant,errors,poses:[]};
  try{
   await page.goto(viewer);await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});await page.waitForFunction(()=>window.getReviewState?.().renderStats.triangles>0,undefined,{timeout:90000});
   await page.addStyleTag({content:'header,.walk-hint{visibility:hidden!important}'});
   for(const camera of ['00000.jpg','00002.jpg']){
    await page.evaluate(name=>window.setReviewCamera(name),camera);await page.waitForTimeout(1500);
    record.poses.push({camera,state:await page.evaluate(()=>window.getReviewState()),renderer:await page.evaluate(()=>window.auditRenderer())});
    await page.screenshot({path:control+'/'+variant+'-'+camera+'.png',scale:'css'});
   }
  }catch(e){record.error=String(e);}
  report.push(record);await page.close();
 }
 await browser.close();await fs.writeFile(control+'/renderer.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report.map(r=>({variant:r.variant,error:r.error,errors:r.errors,renderer:r.poses[0]?.renderer}))));
})();
