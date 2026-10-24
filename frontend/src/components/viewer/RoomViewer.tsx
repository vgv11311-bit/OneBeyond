import React, { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { SpaceJob } from '../../types';
const UnityViewer=lazy(()=>import('./UnityViewer').then(m=>({default:m.UnityViewer})));
const ThreeViewer=lazy(()=>import('./SpatialViewer').then(m=>({default:m.SpatialViewer})));
const GaussianViewer=lazy(()=>import('./GaussianViewer').then(m=>({default:m.GaussianViewer})));

export const RoomViewer:React.FC<{job:SpaceJob;onExit:()=>void}>=({job,onExit})=>{
  const exportUrl=job.gaussian?.ply_url;
  const canExport=!!exportUrl&&/^\/api\/spaces\/[A-Za-z0-9_-]+\/gaussian\/[a-f0-9]{32}\/scene\.ply$/.test(exportUrl);
  const [photographic,setPhotographic]=useState(job.gaussian?.status==='ready');
  const [photoError,setPhotoError]=useState('');
  const leavePhoto=useCallback((reason:string)=>{setPhotoError(reason);setPhotographic(false);},[]);
  useEffect(()=>{setPhotographic(job.gaussian?.status==='ready');setPhotoError('');},[job.job_id,job.gaussian?.status]);
  const [fallback,setFallback]=useState(()=>new URLSearchParams(location.search).get('renderer')==='three');
  const [reason,setReason]=useState('');
  const [checking,setChecking]=useState(!fallback);
  useEffect(()=>{
    if(fallback) {setChecking(false);return;}
    const controller=new AbortController();
    let active=true;
    const timer=setTimeout(()=>controller.abort(),5000);
    // An unpublished Unity build must never block the working application.
    fetch('/unity/viewer.json',{signal:controller.signal,cache:'no-cache'}).then(async response=>{
      if(!response.ok||!response.headers.get('content-type')?.includes('json')) throw new Error('Not published');
      const manifest=await response.json();if(manifest.version!==1)throw new Error('Invalid manifest');
    }).catch(()=>{if(active)setFallback(true);})
      .finally(()=>{clearTimeout(timer);if(active)setChecking(false);});
    return ()=>{active=false;clearTimeout(timer);controller.abort();};
  },[fallback]);
  if(photographic)return <Suspense fallback={<div role="status" className="p-8 text-white">Loading photographic viewer…</div>}><GaussianViewer job={job} onExit={onExit} onFallback={leavePhoto}/></Suspense>;
  if(!job.glb_url)return <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950 p-6 text-white"><div className="max-w-md space-y-4"><p role="alert">{photoError || 'The photographic viewer is unavailable on this device.'}</p><p>This capture has no separate editable room.</p><div className="flex gap-4"><button onClick={()=>{setPhotoError('');setPhotographic(true);}}>Retry viewer</button>{canExport && <a href={exportUrl} download={`${job.job_id}.ply`}>Download splat</a>}<button onClick={onExit}>Exit room</button></div></div></div>;
  if(checking)return <div role="status" className="p-8 text-white">Checking 3D viewer…</div>;
  return <Suspense fallback={<div role="status" className="p-8 text-white">Loading 3D viewer…</div>}>
    {job.gaussian?.status==='ready'&&<button className="fixed right-4 top-20 z-[60] rounded bg-black/80 p-3 text-white" onClick={()=>setPhotographic(true)}>Photographic room</button>}
    {(photoError||job.gaussian?.reason)&&<div role="status" className="fixed bottom-24 left-4 z-[60] max-w-sm rounded bg-black/80 p-3 text-xs text-slate-200">{photoError||job.gaussian?.reason}</div>}
    {fallback?<><ThreeViewer job={job} onExit={onExit}/>{reason&&<div role="status" className="fixed bottom-16 left-4 z-50 max-w-sm rounded bg-black/80 p-3 text-xs text-slate-200">Unity unavailable: {reason} Using the existing viewer.</div>}</>
    :<UnityViewer key={job.job_id} job={job} onExit={onExit} onFallback={reason=>{setReason(reason);setFallback(true);}}/>}
  </Suspense>;
};
