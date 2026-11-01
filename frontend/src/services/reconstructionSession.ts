import { SpaceJob } from '../types';

export async function resumeActiveReconstruction(storage:Pick<Storage,'getItem'|'removeItem'>,getStatus:(id:string)=>Promise<SpaceJob>):Promise<SpaceJob|null>{
  const raw=storage.getItem('activeReconstruction');
  if(!raw)return null;
  let active:{job_id:string;attempt_id:string};
  try{
    active=JSON.parse(raw);
    if(!active||typeof active.job_id!=='string'||!/^[\w-]+$/.test(active.job_id)||typeof active.attempt_id!=='string')throw new Error('Invalid saved job');
  }catch{storage.removeItem('activeReconstruction');return null;}
  // Read status only. Never restart or cancel work as a side effect of reload.
  const job=await getStatus(active.job_id);
  if(job.attempt_id!==active.attempt_id){storage.removeItem('activeReconstruction');return null;}
  if(job.status!=='processing')storage.removeItem('activeReconstruction');
  return job;
}
