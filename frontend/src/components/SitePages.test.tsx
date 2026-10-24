import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { SpaceJob } from '../types';
import { SitePages } from './SitePages';

const space = (index: number) => ({ job_id: `hall-${index}`, name: `Hall ${index}`, status: 'completed' }) as SpaceJob;
const render = (status: 'loading' | 'ready' | 'error', recentSpaces: SpaceJob[], spacesError = '') =>
  renderToStaticMarkup(<SitePages route="recent" onOpenUpload={vi.fn()} onExploreSample={vi.fn()} onSelectSpace={vi.fn()}
    recentSpaces={recentSpaces} spacesStatus={status} spacesError={spacesError} onRetrySpaces={vi.fn()} />);

describe('Your Spaces collection', () => {
  it('labels review results without treating completed processing as quality acceptance', () => {
    const review = { ...space(1), gaussian: { status: 'ready', requires_review: true, quality_accepted: false } } as SpaceJob;
    expect(render('ready', [review])).toContain('Review required · quality not accepted');
    expect(render('ready', [space(2)])).not.toContain('quality accepted');
    expect(render('ready', [{ ...space(3), status: 'review_required' }])).toContain('Review required');
  });
  it('does not show the empty state during first load or after an error', () => {
    expect(render('loading', [])).toContain('Loading saved spaces');
    expect(render('loading', [])).not.toContain('Your collection begins here');
    expect(render('error', [], 'HTTP 503')).toContain('Retry');
    expect(render('error', [], 'HTTP 503')).not.toContain('Your collection begins here');
    expect(render('ready', [])).toContain('Your collection begins here');
  });
  it('preserves visible cards on refresh error', () => {
    const html = render('error', [space(1), space(2)], 'HTTP 503');
    expect(html).toContain('Showing 2 of 2 saved spaces');
    expect(html).toContain('data-testid="space-hall-1"');
    expect(html).toContain('data-testid="space-hall-2"');
    expect(html).toContain('Showing the last loaded collection');
  });
  it('reports the visible card count rather than claiming all cards are shown', () => {
    const html = render('ready', Array.from({ length: 13 }, (_, index) => space(index)));
    expect(html).toContain('Showing 12 of 13 saved spaces');
    expect((html.match(/data-testid="space-hall-/g) || []).length).toBe(12);
    expect(html).toContain('Show more spaces');
  });
});
