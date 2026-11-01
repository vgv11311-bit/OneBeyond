import { expect,it,vi } from 'vitest';
import { resumeActiveReconstruction } from './reconstructionSession';
import { SpaceJob } from '../types';

it('reload reconnects to the same running attempt without creating another job',async()=>{
  const storage={getItem:()=>JSON.stringify({job_id:'room-1',attempt_id:'attempt-1'}),removeItem:vi.fn()};
  const job={job_id:'room-1',attempt_id:'attempt-1',status:'processing'} as SpaceJob;
  const getStatus=vi.fn().mockResolvedValue(job);
  expect(await resumeActiveReconstruction(storage,getStatus)).toBe(job);
  expect(getStatus).toHaveBeenCalledExactlyOnceWith('room-1');expect(storage.removeItem).not.toHaveBeenCalled();
});
it('a connection failure retains the saved attempt for another reconnect',async()=>{
  const storage={getItem:()=>JSON.stringify({job_id:'room-1',attempt_id:'attempt-1'}),removeItem:vi.fn()};
  await expect(resumeActiveReconstruction(storage,async()=>{throw new Error('offline');})).rejects.toThrow('offline');
  expect(storage.removeItem).not.toHaveBeenCalled();
});
it('cancelled work is restored as cancelled, never restarted',async()=>{
  const storage={getItem:()=>JSON.stringify({job_id:'room-1',attempt_id:'attempt-1'}),removeItem:vi.fn()};
  const job={job_id:'room-1',attempt_id:'attempt-1',status:'cancelled'} as SpaceJob;
  expect((await resumeActiveReconstruction(storage,async()=>job))?.status).toBe('cancelled');
  expect(storage.removeItem).toHaveBeenCalledWith('activeReconstruction');
});
