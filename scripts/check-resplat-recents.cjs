const {chromium}=require('playwright');
const fs=require('node:fs/promises');
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',error=>errors.push(String(error)));
 const report={errors,qualityAccepted:false};
 try{
  const payload=await(await page.request.get('http://127.0.0.1:5000/api/spaces')).json();
  const card=payload.spaces.find(space=>space.job_id==='bf37e7eb-f8f');
  if(!card||card.status!=='review_required'||card.gaussian.quality_accepted!==false)throw Error('Missing labelled review card');
  report.card=card;
  await page.goto('http://127.0.0.1:5000/spaces');
  await page.getByTestId('space-bf37e7eb-f8f').click();
  const frame=page.frameLocator('iframe[title="Photographic room reconstruction"]');
  await frame.locator('canvas').waitFor({timeout:90000});
  await page.getByText('Saved in Recents for review, not quality approved.',{exact:false}).waitFor();
  const child=page.frames().find(f=>f.url().includes('/bf37e7eb-f8f/gaussian/'));
  await child.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  await child.waitForFunction(()=>window.getReviewState?.().renderStats.triangles>0,undefined,{timeout:90000});
  await page.waitForTimeout(1200);
  report.before=await child.evaluate(()=>window.getReviewState());
  report.constrained=await child.evaluate(()=>{
   const before=window.getReviewState().position;window.tryReviewMove([before[0]+100,before[1],before[2]]);
   return window.getReviewState();
  });
  if(!report.before.enclosure||!report.before.collision||errors.length)throw Error('Viewer support missing');
  const p=report.constrained.position,b=report.before.collision.bounds;
  if(p.some((v,i)=>!Number.isFinite(v)||v<b.min[i]||v>b.max[i]))throw Error('Escaped enclosure');
  await page.screenshot({path:'reports/resplat-oct24/recents-review.png'});
  report.openedFromRecents=true;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile('reports/resplat-oct24/recents-review.json',JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report));
})();
