import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { ViewerErrorBoundary } from './ViewerErrorBoundary';

describe('viewer failure recovery', () => {
  it('keeps the saved reconstruction and exposes an explicit retry', () => {
    const boundary = new ViewerErrorBoundary({onRetry:vi.fn()});
    boundary.state = ViewerErrorBoundary.getDerivedStateFromError();
    const markup = renderToStaticMarkup(boundary.render());
    expect(markup).toContain('role="alert"');
    expect(markup).toContain('Retry loading model');
    expect(markup).toContain('reconstruction is still saved');
  });
});
