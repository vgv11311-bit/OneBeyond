// Real saved artifact moving-view inspection. Never invents a quality pass.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path');
(async()=>{
 const pointer=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
 const out=path.resolve(process.argv[3]);await fs.mkdir(out,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>requests.push(r.url()));
 const report={job:pointer.job_id,errors,qualityAccepted:false,scale:'COLMAP SfM units, not measured metres',poses:[]};
 try{
  const state=await(await page.request.get('http://127.0.0.1:5000/api/reconstruct/status/'+pointer.job_id)).json();
  const config=await(await page.request.get('http://127.0.0.1:5000'+state.gaussian.config_url)).json();
  if(config.surface||config.surfaces||config.fused_surface||!config.native_splat_display)throw Error('Expected native joint Gaussians, not photo sheets');
  await page.goto('http://127.0.0.1:5000'+pointer.viewer_url);
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  const error=await page.evaluate(()=>window.reviewError);if(error)throw Error(error);
  const anchor=config.cameras[0].name;
  const spread=Math.max(...config.cameras.map(c=>Math.hypot(...c.position.map((v,i)=>v-config.cameras[0].position[i]))));
  report.cameraSpread=spread;
  for(const [label,offset,yaw,pitch] of [
   ['entrance',[0,0,0],0,0],['left',[-spread*.15,0,0],0,0],
   ['right',[spread*.15,0,0],0,0],['forward',[0,0,-spread*.2],0,0],
   ['rear',[0,0,0],Math.PI,0],['ceiling',[0,0,0],0,1.1]]){
   await page.evaluate(({anchor,offset,yaw,pitch})=>{window.setReviewCamera(anchor);const p=window.getReviewState().position;window.tryReviewMove(p.map((v,i)=>v+offset[i]));window.turnReviewCamera(yaw);window.pitchReviewCamera(pitch);},{anchor,offset,yaw,pitch});
   await page.waitForTimeout(1800);
   report.poses.push({label,state:await page.evaluate(()=>window.getReviewState())});
   await page.screenshot({path:path.join(out,label+'.png')});
  }
  report.rawPlyRequested=requests.some(u=>/\/scene\.ply(?:\?|$)/.test(u));
  report.photoMeshesRequested=requests.some(u=>/surface.*\.glb/.test(u));
  if(!report.rawPlyRequested||report.photoMeshesRequested||errors.length)throw Error('Native renderer failed');
  report.rendererChecksPassed=true;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(out,'review.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report));
})();
