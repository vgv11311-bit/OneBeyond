const {chromium}=require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');const path=require('node:path');
(async()=>{
 const job=process.argv[2];if(!/^[\w-]+$/.test(job||''))throw new Error('Provide job ID');
 const output=process.argv[3]||'video-intake-oct21';if(!/^[\w-]+$/.test(output))throw new Error('Invalid report directory');
 const directory=path.resolve(__dirname,'../reports',output);await fs.mkdir(directory,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1000,height:700}});
 const report={job,errors:[]};page.on('pageerror',e=>report.errors.push(String(e)));
 try{
  const state=await (await page.request.get('http://127.0.0.1:5000/api/reconstruct/status/'+job)).json();
  await page.goto('http://127.0.0.1:5000'+state.gaussian.viewer_url);
  await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  const error=await page.evaluate(()=>window.reviewError);if(error)throw new Error(error);
  await page.evaluate(()=>window.turnReviewCamera(Math.PI));await page.waitForTimeout(1000);
  report.view=await page.evaluate(()=>window.getReviewState());
  if(!report.view.enclosureOnly)throw new Error('Unsupported rear-facing view still renders floaters');
  report.pixels=await page.evaluate(()=>{
   const source=document.querySelector('canvas'),copy=document.createElement('canvas');
   copy.width=source.width;copy.height=source.height;const context=copy.getContext('2d');context.drawImage(source,0,0);
   const {data}=context.getImageData(0,0,copy.width,copy.height);let dark=0,sum=0;
   for(let i=0;i<data.length;i+=4){const v=(data[i]+data[i+1]+data[i+2])/3;sum+=v;if(v<12)dark++;}
   const channels=[[],[],[]];
   for(let y=Math.floor(copy.height*.35);y<copy.height*.6;y++)for(let x=Math.floor(copy.width*.4);x<copy.width*.6;x++){
    const i=4*(y*copy.width+x);for(let c=0;c<3;c++)channels[c].push(data[i+c]);
   }
   const centerRange=channels.map(values=>Math.max(...values)-Math.min(...values));
   return {darkFraction:dark/(data.length/4),brightness:sum/(data.length/4),centerRange};
  });
  if(report.pixels.darkFraction>.01)throw new Error('Rear enclosure has dark void pixels');
  if(Math.max(...report.pixels.centerRange)>2)throw new Error('Rear wall is a gradient or photographic smear, not a plain surface');
  await page.screenshot({path:path.join(directory,'rear-enclosure.png')});
  const stopped=await page.evaluate(()=>window.tryReviewMove([0,0,100]));
  report.stopped=stopped;
  if(stopped[2]>report.view.collision.bounds.max[2]-report.view.collision.radius+1e-5)throw new Error('Can escape rear wall');
  if(report.errors.length)throw new Error(report.errors.join(';'));
  report.passed=true;
 }catch(error){report.error=String(error);process.exitCode=1;}
 finally{await fs.writeFile(path.join(directory,'enclosure-check.json'),JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify(report,null,2));
})();
