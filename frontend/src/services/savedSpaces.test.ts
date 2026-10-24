import { describe, expect, it, vi } from 'vitest';
import { SpaceJob } from '../types';
import { readSavedSpacesCache, SavedSpacesController, SavedSpacesState } from './savedSpaces';

const hall = (id: string) => ({ job_id: id, name: id, status: 'review_required' }) as SpaceJob;
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

describe('saved spaces refresh', () => {
  it('distinguishes a slow first load from a confirmed empty collection', async () => {
    const request = deferred<SpaceJob[]>();
    const publish = vi.fn();
    const controller = new SavedSpacesController(() => request.promise, publish);
    const refresh = controller.refresh();
    expect(controller.snapshot).toEqual({ spaces: [], status: 'loading', error: '' });
    request.resolve([]);
    await refresh;
    expect(controller.snapshot).toEqual({ spaces: [], status: 'ready', error: '' });
  });

  it('preserves the last successful list on refresh failure and allows retry', async () => {
    const prior = [hall('hall-audit'), hall('hall-resplat')];
    const requests = [deferred<SpaceJob[]>(), deferred<SpaceJob[]>(), deferred<SpaceJob[]>()];
    const cache = new Map<string, string>();
    const storage = { getItem: (key: string) => cache.get(key) ?? null, setItem: (key: string, value: string) => { cache.set(key, value); } };
    let index = 0;
    const controller = new SavedSpacesController(() => requests[index++].promise, vi.fn(), storage);
    const first = controller.refresh();
    requests[0].resolve(prior);
    await first;
    expect(readSavedSpacesCache(storage)).toEqual(prior);
    const failed = controller.refresh();
    expect(controller.snapshot.spaces).toEqual(prior);
    expect(controller.snapshot.status).toBe('loading');
    requests[1].reject(new Error('HTTP 503'));
    await failed;
    expect(controller.snapshot).toEqual({ spaces: prior, status: 'error', error: 'HTTP 503' });
    expect(readSavedSpacesCache(storage)).toEqual(prior);
    const retried = controller.refresh();
    requests[2].resolve(prior);
    await retried;
    expect(controller.snapshot.status).toBe('ready');
  });

  it('ignores an older response after a newer refresh completes', async () => {
    const firstRequest = deferred<SpaceJob[]>();
    const secondRequest = deferred<SpaceJob[]>();
    let index = 0;
    const controller = new SavedSpacesController(() => [firstRequest.promise, secondRequest.promise][index++], vi.fn());
    const first = controller.refresh();
    const second = controller.refresh();
    secondRequest.resolve([hall('newer')]);
    await second;
    firstRequest.resolve([]);
    await first;
    expect(controller.snapshot.spaces.map(space => space.job_id)).toEqual(['newer']);
  });

  it('ignores an older failure after a newer refresh succeeds', async () => {
    const firstRequest = deferred<SpaceJob[]>();
    const secondRequest = deferred<SpaceJob[]>();
    let index = 0;
    const states: SavedSpacesState[] = [];
    const controller = new SavedSpacesController(() => [firstRequest.promise, secondRequest.promise][index++], state => states.push(state));
    const first = controller.refresh();
    const second = controller.refresh();
    secondRequest.resolve([hall('newer')]);
    await second;
    firstRequest.reject(new Error('old network error'));
    await first;
    expect(controller.snapshot.status).toBe('ready');
    expect(states.at(-1)?.spaces[0].job_id).toBe('newer');
  });
});
