import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
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
    const result = expect(api.getJobStatus('room-1')).rejects.toThrow('aborted');
    await vi.advanceTimersByTimeAsync(20000);
    await result;
    expect(vi.getTimerCount()).toBe(0);
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
