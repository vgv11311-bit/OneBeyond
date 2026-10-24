// Same explicit path, no reconstruction or source writes. Real W is tested
// separately from pose replay so actual keyboard motion cannot be inferred.
const {chromium}=require('playwright'),fs=require('node:fs/promises');
(async()=>{
 const root='reports/hall-stage-audit-oct24',manifest=JSON.parse(await fs.readFile(root+'/manifest.json','utf8'));
 const browser=await chromium.launch({channel:'chrome',headless:true}),report=[];
 for(const [job,item] of Object.entries(manifest.scenes)){
  const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1}),errors=[],requests=[];page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>requests.push(r.url()));
  await page.route('**/viewer',async route=>{
   const response=await route.fetch();let html=await response.text();
   html=html.replace('window.reviewReady=true;',`window.stageAudit=()=>({css:[canvas.clientWidth,canvas.clientHeight],buffer:renderer.getDrawingBufferSize(new THREE.Vector2()).toArray(),dpr:devicePixelRatio,ratio:renderer.getPixelRatio(),antialias:renderer.getContext().getContextAttributes().antialias,samples:renderer.getContext().getParameter(renderer.getContext().SAMPLES),activeTarget:!!renderer.getRenderTarget(),native:config.native_splat_display,photoMesh:hasPhotoMesh,three:THREE.REVISION,source:model?.splats.constructor.name,spark:{active:spark?.activeSplats,blur:spark?.blurAmount,preBlur:spark?.preBlurAmount,aperture:spark?.apertureAngle,lod:spark?.enableLod,depthTest:spark?.material.depthTest},projection:camera.projectionMatrix.toArray()});window.reviewReady=true;`);
   await route.fulfill({response,body:html});
  });
  const result={job,errors,poses:[]};
  try{
   await page.goto('http://127.0.0.1:5000'+item.viewer_url);await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});await page.waitForFunction(()=>window.getReviewState?.().renderStats.triangles>0,undefined,{timeout:90000});
   await page.addStyleTag({content:'header,.walk-hint{visibility:hidden!important}'});
   const start=await page.evaluate(()=>window.getReviewState().position);
   const cases=[['initial',0,[0,0,0]],['forward-moving',0,[-.3,0,-.5]],['forward-settled',0,[-.3,0,-.5]],['right',-.75,[0,0,0]],['near-boundary',0,[0,0,-1]],['return',0,[0,0,0]]];
   for(const [name,yaw,offset] of cases){
    await page.evaluate(({start,yaw,offset})=>{window.setReviewCamera('00000.jpg');window.tryReviewMove(start.map((v,i)=>v+offset[i]));window.turnReviewCamera(yaw);},{start,yaw,offset});
    if(name==='forward-moving')await page.waitForTimeout(30);else await page.waitForTimeout(1400);
    result.poses.push({name,state:await page.evaluate(()=>window.getReviewState()),renderer:await page.evaluate(()=>window.stageAudit())});
    await page.screenshot({path:root+'/'+job+'-'+name+'.png',scale:'css'});
   }
   await page.evaluate(()=>window.setReviewCamera('00000.jpg'));await page.locator('canvas').focus();
   const before=await page.evaluate(()=>window.getReviewState());await page.keyboard.down('KeyW');
   await page.waitForFunction(frames=>window.getReviewState().drawnFrames>=frames+3,before.drawnFrames,{timeout:10000});
   result.actualKeyMoving=await page.evaluate(()=>window.getReviewState());await page.keyboard.up('KeyW');await page.waitForTimeout(1800);result.actualKeySettled=await page.evaluate(()=>window.getReviewState());
   result.wMoved=result.actualKeyMoving.position.some((v,i)=>Math.abs(v-before.position[i])>1e-6);
   if(!result.wMoved)throw Error('Actual W movement was not demonstrated');
   result.plyRequests=requests.filter(r=>r.endsWith('/scene.ply'));result.meshRequests=requests.filter(r=>r.endsWith('.glb'));
  }catch(error){result.error=String(error);}
  report.push(result);await page.close();
 }
 await browser.close();await fs.writeFile(root+'/two-scene-motion.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report.map(r=>({job:r.job,wMoved:r.wMoved,error:r.error,errors:r.errors,renderer:r.poses[0]?.renderer}))));
})();
