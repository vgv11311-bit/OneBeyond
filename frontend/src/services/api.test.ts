import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
describe('existing GPU reconnection',()=>{
  it('finishes within 45 seconds even when fetch ignores abort and allows a retry',async()=>{
    vi.useFakeTimers();vi.stubGlobal('fetch',vi.fn(()=>new Promise(()=>{})));
    const pending=expect(api.reconnectResplat()).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(45000);await pending;
    expect(vi.getTimerCount()).toBe(0);
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({can_run:true})}));
    await api.reconnectResplat();expect(vi.getTimerCount()).toBe(0);
  });
  it('shares one in-flight connection check',async()=>{
    let finish!:(value:unknown)=>void;
    const fetchMock=vi.fn(()=>new Promise(resolve=>{finish=resolve;}));vi.stubGlobal('fetch',fetchMock);
    const first=api.reconnectResplat(),second=api.reconnectResplat();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    finish({ok:true,json:async()=>({can_run:true})});await Promise.all([first,second]);
  });
  it('bounds the follow-up runtime refresh so the spinner cannot hang there',async()=>{
    vi.useFakeTimers();vi.stubGlobal('fetch',vi.fn(()=>new Promise(()=>{})));
    const pending=expect(api.getEngines()).rejects.toThrow('refresh timed out');
    await vi.advanceTimersByTimeAsync(5000);await pending;expect(vi.getTimerCount()).toBe(0);
  });
  it('requests verification through the local backend',async()=>{
    const fetchMock=vi.fn().mockResolvedValue({ok:true,json:async()=>({can_run:true})});
    vi.stubGlobal('fetch',fetchMock);
    await api.reconnectResplat();
    expect(fetchMock).toHaveBeenCalledWith('/api/reconstruction/resplat/connect',expect.objectContaining({method:'POST',signal:expect.any(AbortSignal)}));
  });
  it('keeps an offline or expired connection as an error',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false,json:async()=>({can_run:false,availability_message:'GPU VM is stopped'})}));
    await expect(api.reconnectResplat()).rejects.toThrow('GPU VM is stopped');
  });
});
describe('saved room hydration', () => {
  it('retrieves the complete saved camera and object metadata', async () => {
    const job = { job_id:'room-1', status:'completed', metrics:{extra_details:{
      lighting:{projection_camera:{position:[0,1.2,-.5]}},scene_objects:[{label:'sofa'}],
    }} };
    const fetchMock = vi.fn().mockResolvedValue({ok:true,json:async () => job});
    vi.stubGlobal('fetch',fetchMock);
    expect(await api.getJobStatus('room-1')).toEqual(job);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/reconstruct/status/room-1');
  });
  it('aborts stalled status reads so opening can recover', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch',vi.fn((_url,options) => new Promise((_resolve,reject) => {
      options.signal.addEventListener('abort',() => reject(new Error('aborted')));
    })));
    const result = expect(api.getJobStatus('room-1')).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(8000);
    await result;
    expect(vi.getTimerCount()).toBe(0);
  });
  it('releases status polling even when fetch ignores cancellation',async()=>{
    vi.useFakeTimers();vi.stubGlobal('fetch',vi.fn(()=>new Promise(()=>{})));
    const pending=expect(api.getJobStatus('room-1')).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(8000);await pending;
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({job_id:'room-1',status:'processing'})}));
    expect((await api.getJobStatus('room-1')).status).toBe('processing');
  });
});

describe('saved spaces API', () => {
  it('accepts a genuine empty result', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ spaces: [], count: 0 }) }));
    expect(await api.listSpaces()).toEqual([]);
  });
  it('does not turn HTTP failures into an empty collection', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    await expect(api.listSpaces()).rejects.toThrow('HTTP 503');
  });
  it.each([{ count: 0 }, { spaces: [], count: 1 }, { spaces: [{ job_id: 'x', status: 'completed' }, { job_id: 'x', status: 'completed' }], count: 2 }])('rejects malformed response %j', async payload => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => payload }));
    await expect(api.listSpaces()).rejects.toThrow('malformed');
  });
});
