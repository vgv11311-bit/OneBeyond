const {chromium}=require('playwright'),fs=require('node:fs/promises');
(async()=>{
 const out='reports/quality-queue-oct24',pointer=JSON.parse(await fs.readFile(out+'/candidate-pointer.json','utf8'));
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const report={scenes:[],errors:[]};
 const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1.25});page.on('pageerror',e=>report.errors.push(String(e)));
 await page.goto('http://127.0.0.1:5000/spaces');await page.getByTestId('space-'+pointer.job_id).waitFor({timeout:45000});
 report.recentCard=await page.getByTestId('space-'+pointer.job_id).innerText();
 if(!report.recentCard.includes('quality not accepted'))throw Error('Review label missing');
 await page.screenshot({path:out+'/recents.png'});
 await page.getByTestId('space-'+pointer.job_id).click();
 const frame=page.frameLocator('iframe');await frame.locator('canvas').waitFor({timeout:90000});
 await page.waitForTimeout(1500);report.galleryViewerUrl=page.frames().map(f=>f.url()).find(u=>u.includes('/gaussian/'));
 if(!report.galleryViewerUrl?.includes(pointer.job_id))throw Error('Wrong room opened');
 for(const [name,url] of [['original','/api/spaces/hall-audit-e208e4c3b9/gaussian/7cc3b9e742b54c8bbc7678689bcd9d02/viewer'],['candidate',pointer.viewer_url]]){
  await page.goto('http://127.0.0.1:5000'+url);await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
  if(await page.evaluate(()=>window.reviewError))throw Error(await page.evaluate(()=>window.reviewError));
  await page.addStyleTag({content:'header,.walk-hint{visibility:hidden!important}'});
  const scene={name,poses:[]};const initial=await page.evaluate(()=>window.getReviewState().position);
  for(const [pose,yaw,delta] of [['initial',0,0],['translated',0,.5],['rotated',-.75,.5],['boundary',0,1]]){
   await page.evaluate(({initial,yaw,delta})=>{window.setReviewCamera('00000.jpg');window.tryReviewMove(initial.map((v,i)=>v+(i===2?-delta:0)));window.turnReviewCamera(yaw);},{initial,yaw,delta});
   await page.waitForTimeout(1500);scene.poses.push({pose,state:await page.evaluate(()=>window.getReviewState())});await page.screenshot({path:out+'/'+name+'-'+pose+'.png',scale:'css'});
  }
  await page.evaluate(()=>window.setReviewCamera('00000.jpg'));await page.locator('canvas').focus();
  const before=await page.evaluate(()=>window.getReviewState());await page.keyboard.down('KeyW');await page.waitForFunction(frames=>window.getReviewState().drawnFrames>=frames+3,before.drawnFrames,{timeout:30000});
  scene.moving=await page.evaluate(()=>window.getReviewState());await page.screenshot({path:out+'/'+name+'-key-moving.png',scale:'css'});await page.keyboard.up('KeyW');await page.waitForTimeout(1800);
  scene.settled=await page.evaluate(()=>window.getReviewState());scene.wMoved=before.position.some((v,i)=>Math.abs(v-scene.moving.position[i])>1e-6);
  scene.buffer=await page.locator('canvas').evaluate(e=>({physical:[e.width,e.height],css:[e.clientWidth,e.clientHeight],dpr:devicePixelRatio}));
  report.scenes.push(scene);await fs.writeFile(out+'/actual-room-browser.json',JSON.stringify(report,null,2));
  if(!scene.wMoved)throw Error(name+' W did not move');
 }
 await fs.writeFile(out+'/actual-room-browser.json',JSON.stringify(report,null,2));await browser.close();console.log(JSON.stringify({errors:report.errors,recentCard:report.recentCard,galleryViewerUrl:report.galleryViewerUrl,scenes:report.scenes.map(s=>({name:s.name,wMoved:s.wMoved,buffer:s.buffer,renderer:s.poses[0].state.nativeRenderer}))}));if(report.errors.length)process.exitCode=1;
})();
