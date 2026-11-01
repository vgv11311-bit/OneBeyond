import { SpaceJob } from '../types';
import { parseSpacesResponse } from './api';

const CACHE_KEY = 'oneBeyond.savedSpaces.v1';

export type SavedSpacesState = {
  spaces: SpaceJob[];
  status: 'loading' | 'ready' | 'error';
  error: string;
};

type Cache = Pick<Storage, 'getItem' | 'setItem'>;

export function readSavedSpacesCache(cache?: Cache): SpaceJob[] {
  try {
    const raw = cache?.getItem(CACHE_KEY);
    return raw ? parseSpacesResponse(JSON.parse(raw)) : [];
  } catch {
    return [];
  }
}

export class SavedSpacesController {
  private generation = 0;
  private state: SavedSpacesState;

  constructor(
    private readonly load: () => Promise<SpaceJob[]>,
    private readonly publish: (state: SavedSpacesState) => void,
    private readonly cache?: Cache,
  ) {
    this.state = { spaces: readSavedSpacesCache(cache), status: 'loading', error: '' };
  }

  get snapshot(): SavedSpacesState { return this.state; }

  async refresh(): Promise<void> {
    const request = ++this.generation;
    this.update({ ...this.state, status: 'loading', error: '' });
    try {
      const spaces = await this.load();
      if (request !== this.generation) return;
      // Only a validated successful response can replace the last good list.
      const validated = parseSpacesResponse({ spaces, count: spaces.length });
      this.update({ spaces: validated, status: 'ready', error: '' });
      try { this.cache?.setItem(CACHE_KEY, JSON.stringify({ spaces: validated, count: validated.length })); }
      catch { /* Storage is optional; the in-memory list remains authoritative. */ }
    } catch (error) {
      if (request !== this.generation) return;
      const message = error instanceof Error ? error.message : 'Could not load saved spaces.';
      this.update({ ...this.state, status: 'error', error: message });
    }
  }

  invalidate(): void { ++this.generation; }

  private update(next: SavedSpacesState): void {
    this.state = next;
    this.publish(next);
  }
}
