// Synthetic front/behind control only, not a claim about real hall floaters.
const {chromium}=require('playwright'),fs=require('node:fs/promises');
(async()=>{
 const root='reports/hall-stage-audit-oct24',out='reports/quality-queue-oct24/native-float';
 const points=JSON.parse(await fs.readFile(root+'/synthetic-wall-depth.json','utf8')).points;
 const config=JSON.parse(await fs.readFile(root+'/baseline/hall-audit-e208e4c3b9/config.json','utf8'));config.native_renderer='native-float-v1';
 const browser=await chromium.launch({channel:'chrome',headless:true}),page=await browser.newPage({viewport:{width:960,height:704}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));await page.route('**/config.json',r=>r.fulfill({json:config}));
 await page.route('**/scene.ply',r=>r.fulfill({path:root+'/synthetic-wall-depth.ply',contentType:'application/octet-stream'}));
 await page.route('**/viewer',async r=>{const response=await r.fetch();await r.fulfill({response,body:(await response.text()).replace('window.reviewReady=true;',`window.projectProbe=p=>{const v=new THREE.Vector3(...p).project(camera);return [(v.x+1)*innerWidth/2,(1-v.y)*innerHeight/2,v.z];};window.reviewReady=true;`)});});
 await page.goto('http://127.0.0.1:5000/api/spaces/hall-audit-e208e4c3b9/gaussian/7cc3b9e742b54c8bbc7678689bcd9d02/viewer');
 await page.waitForFunction(()=>window.reviewReady||window.reviewError,undefined,{timeout:90000});
 if(await page.evaluate(()=>window.reviewError))throw Error(await page.evaluate(()=>window.reviewError));
 await page.evaluate(()=>{window.setReviewCamera('00000.jpg');window.turnReviewCamera(Math.PI);});await page.waitForTimeout(1500);
 await page.addStyleTag({content:'header,.walk-hint{visibility:hidden!important}'});
 const locations=await page.evaluate(points=>points.map(p=>window.projectProbe(p)),points);
 await page.screenshot({path:out+'/wall-off.png'});
 await page.locator('input[type=checkbox]').evaluate(input=>{input.checked=true;input.dispatchEvent(new Event('change',{bubbles:true}));});await page.waitForTimeout(1000);
 await page.screenshot({path:out+'/wall-on.png'});
 await fs.writeFile(out+'/wall.json',JSON.stringify({errors,locations,state:await page.evaluate(()=>window.getReviewState())},null,2));
 await browser.close();console.log(JSON.stringify({errors,locations}));if(errors.length)process.exitCode=1;
})();
