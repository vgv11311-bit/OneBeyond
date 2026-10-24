// Same saved geometry/cameras, archived pre-optimization viewer versus current.
const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
(async()=>{
 const job=process.argv[2],output=process.argv[3];if(!/^[\w-]+$/.test(job||'')||!/^[\w-]+$/.test(output||''))throw Error('Job/report required');
 const directory=path.resolve(__dirname,'../reports',output);await fs.mkdir(directory,{recursive:true});
 const archive=path.resolve(__dirname,'../reports/shared-release/photographic-20261022T044437Z.tar.gz');
 const old=execFileSync('tar',['-xOf',archive,'scripts/splat-review.html'],{encoding:'utf8'});
 const browser=await chromium.launch({channel:'chrome',headless:true}),report={job,archive,qualityAcceptance:false,runs:[],comparisons:[],pixelTolerance:'At most 2 channel levels on fewer than 0.01% of pixels; records actual differences and strict equality separately (cross-context MSAA rounding).'};
 const baselinePixels=new Map();
 try{
  for(const baseline of [true,false]){
   const page=await browser.newPage({viewport:{width:1000,height:700}}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
   const state=await(await page.request.get('http://127.0.0.1:5000/api/reconstruct/status/'+job)).json(),prefix=state.gaussian.viewer_url.replace(/\/viewer$/,'');
   if(baseline)await page.route('**'+state.gaussian.viewer_url,route=>route.fulfill({body:old.replaceAll('/vendor/','/api/gaussian-vendor/').replace("'/config.json'",JSON.stringify(prefix+'/config.json')).replace("'/scene.ply'",JSON.stringify(prefix+'/scene.ply')),contentType:'text/html'}));
   await page.goto('http://127.0.0.1:5000'+state.gaussian.viewer_url);
   await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
   if(await page.evaluate(()=>window.reviewError))throw Error(await page.evaluate(()=>window.reviewError));
   const name=await page.locator('#view').inputValue(),run={baseline,poses:[],errors};
   for(const [label,yaw,pitch,offset] of [['entrance',0,0,0],['translated',0,0,.12],['rear',Math.PI,0,0],['ceiling',0,1.1,0]]){
    await page.evaluate(({name,yaw,pitch,offset})=>{window.setReviewCamera(name);if(offset)window.moveReviewCamera(offset);window.turnReviewCamera(yaw);window.pitchReviewCamera(pitch);},{name,yaw,pitch,offset});
    await page.waitForTimeout(250);const before=await page.evaluate(()=>window.getReviewState());await page.waitForTimeout(500);
    const after=await page.evaluate(()=>window.getReviewState()),pixels=await page.evaluate(()=>{
     const canvas=document.querySelector('canvas'),copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;
     const context=copy.getContext('2d');context.drawImage(canvas,0,0);const bytes=context.getImageData(0,0,copy.width,copy.height).data;
     let text='';for(let i=0;i<bytes.length;i+=32768)text+=String.fromCharCode(...bytes.subarray(i,i+32768));return btoa(text);
    });
    const raw=Buffer.from(pixels,'base64');
    if(baseline)baselinePixels.set(label,raw);
    else{
     const prior=baselinePixels.get(label);if(prior.length!==raw.length)throw Error('Canvas extent changed');
     let changed=0,max=0,total=0;for(let i=0;i<raw.length;i+=4){let pixel=0;for(let c=0;c<4;c++){const d=Math.abs(prior[i+c]-raw[i+c]);max=Math.max(max,d);total+=d;pixel=Math.max(pixel,d);}if(pixel)changed++;}
     report.comparisons.push({label,strictlyIdentical:changed===0,changedPixels:changed,changedFraction:changed/(raw.length/4),maxChannelDifference:max,meanChannelDifference:total/raw.length,
      equivalentWithinTolerance:max<=2&&changed/(raw.length/4)<.0001&&total/raw.length<.001});
    }
    run.poses.push({label,pixelSha256:crypto.createHash('sha256').update(pixels).digest('hex'),renderCallsDuringIdle:after.renderStats.frame-before.renderStats.frame});
    if(label==='entrance')await page.screenshot({path:path.join(directory,baseline?'before.png':'after.png')});
   }
   if(errors.length)throw Error(errors.join(';'));report.runs.push(run);await page.close();
  }
  report.matches=report.runs[0].poses.map((p,i)=>({label:p.label,identicalPixels:p.pixelSha256===report.runs[1].poses[i].pixelSha256,
    beforeIdleRenderCalls:p.renderCallsDuringIdle,afterIdleRenderCalls:report.runs[1].poses[i].renderCallsDuringIdle}));
  report.passed=report.matches.every((p,i)=>report.comparisons[i].equivalentWithinTolerance&&p.afterIdleRenderCalls===0&&p.beforeIdleRenderCalls>0);
  if(!report.passed)process.exitCode=1;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(directory,'comparison.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify({passed:report.passed,matches:report.matches,comparisons:report.comparisons,error:report.error},null,2));
})();
