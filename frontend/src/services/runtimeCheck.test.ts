import { afterEach, expect, it, vi } from 'vitest';
import { checkRuntimeWithDeadline } from './runtimeCheck';

afterEach(() => vi.useRealTimers());
it('releases a stuck UI callback and permits the next check', async () => {
  vi.useFakeTimers();
  const waiting = expect(checkRuntimeWithDeadline(() => new Promise(() => {}))).rejects.toThrow('Loading has stopped');
  await vi.advanceTimersByTimeAsync(50000);
  await waiting;
  await checkRuntimeWithDeadline(async () => {});
  expect(vi.getTimerCount()).toBe(0);
});
it('preserves a connection failure and clears its deadline', async () => {
  vi.useFakeTimers();
  await expect(checkRuntimeWithDeadline(async () => { throw new Error('VM is stopped'); })).rejects.toThrow('VM is stopped');
  expect(vi.getTimerCount()).toBe(0);
});
