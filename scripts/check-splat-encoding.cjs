const {chromium}=require('playwright'),fs=require('node:fs/promises');
(async()=>{
 const root='reports/hall-stage-audit-oct24',control=root+'/archived-control';const audit=JSON.parse(await fs.readFile(root+'/encoding-audit.json','utf8')).control;
 const browser=await chromium.launch({channel:'chrome',headless:true}),report=[];
 for(const variant of ['default-verified','rgb-wide-verified','rgb-sh-wide-verified','rgb-sh-scale-wide-verified','rgb-sh-scale-wide-accum-verified','extended-accum-verified']){
  const page=await browser.newPage({viewport:{width:960,height:704},deviceScaleFactor:1}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const encoding={};if(!variant.startsWith('default'))for(const k of ['rgbMin','rgbMax'])encoding[k]=audit.encoding[k];if(variant.includes('sh'))for(const k of ['sh1Max','sh2Max','sh3Max'])encoding[k]=audit.encoding[k];if(variant.includes('scale'))for(const k of ['lnScaleMin','lnScaleMax'])encoding[k]=audit.encoding[k];
  await page.route('**/config.json',r=>r.fulfill({path:control+'/config.json',contentType:'application/json'}));await page.route('**/scene.ply',r=>r.fulfill({path:control+'/scene-fixed.ply',contentType:'application/octet-stream'}));
  await page.route('**/viewer',async route=>{
   const res=await route.fetch();let html=await res.text();
   let replaced=0;html=html.replace(/new SplatMesh\(\{url:('[^']*'|"[^"]*")\}\)/,(_,url)=>{replaced++;return `new SplatMesh({url:${url},${variant.startsWith('extended')?'extSplats:true':'splatEncoding:'+JSON.stringify(encoding)}})`;});
   if(replaced!==1)throw Error('Model variant did not apply');
   if(variant.includes('accum'))html=html.replace('depthTest:!!(config.native_splat_display&&config.navigation)','accumExtSplats:true,depthTest:false');
   html=html.replace('renderer.setClearColor(0x171b20,1)','renderer.setClearColor(0,1)');
   html=html.replace('window.reviewReady=true;',`window.decodedProbe=indices=>({type:model.splats.constructor.name,encoding:model.splats.splatEncoding,accumExt:spark.accumExtSplats,samples:indices.map(i=>{const s=model.splats.getSplat(i);return {index:i,center:s.center.toArray(),scales:s.scales.toArray(),color:s.color.toArray(),opacity:s.opacity};})});window.reviewReady=true;`);
   await route.fulfill({response:res,body:html});
  });
  const record={variant,encoding,errors};
  try{await page.goto('http://127.0.0.1:5000/api/spaces/bf37e7eb-f8f/gaussian/4fec4f3c7ac94b1a98c394210c5d6012/viewer');await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});await page.waitForFunction(()=>window.getReviewState?.().renderStats.triangles>0,undefined,{timeout:90000});await page.waitForTimeout(1500);await page.addStyleTag({content:'header,.walk-hint{visibility:hidden!important}'});await page.screenshot({path:control+'/encoding-'+variant+'.png',scale:'css'});record.decoded=await page.evaluate(indices=>window.decodedProbe(indices),audit.samples.map(s=>s.index));}
  catch(e){record.error=String(e);}
  report.push(record);await page.close();
 }
 await browser.close();await fs.writeFile(root+'/encoding-verified-browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
})();
