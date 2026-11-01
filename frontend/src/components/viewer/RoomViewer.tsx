import React,{useState,useCallback,useEffect} from 'react';
import {SpaceJob} from '../../types';
import {GaussianViewer} from './GaussianViewer';
export const RoomViewer:React.FC<{job:SpaceJob;onExit:()=>void}>=({job,onExit})=>{
  const [error,setError]=useState('');
  const failed=useCallback((message:string)=>setError(message),[]);
  useEffect(()=>setError(''),[job.job_id]);
  if(error||job.gaussian?.status!=='ready')return <div className="fixed inset-0 z-50 bg-slate-950 text-white flex items-center justify-center">
    <div><p role="alert">{error||'This room has no ready Gaussian output.'}</p>
    <button onClick={()=>setError('')}>Retry viewer</button><button onClick={onExit}>Exit room</button></div></div>;
  return <GaussianViewer job={job} onExit={onExit} onFallback={failed}/>;
};
