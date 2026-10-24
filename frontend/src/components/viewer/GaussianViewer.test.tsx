import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { GaussianViewer } from './GaussianViewer';
import { SpaceJob } from '../../types';

const job = {job_id:'room',glb_url:'/api/spaces/room/model.glb',gaussian:{status:'ready',viewer_url:'/api/spaces/room/gaussian/'+'a'.repeat(32)+'/viewer'}} as SpaceJob;

describe('photographic viewer contract', () => {
  it('keeps a saved experimental review distinct from quality approval', () => {
    const markup=renderToStaticMarkup(<GaussianViewer job={{...job,engine_id:'resplat_experimental',status:'review_required',gaussian:{...job.gaussian!,requires_review:true,quality_accepted:false,saved_to_recents:true}}} onExit={vi.fn()} onFallback={vi.fn()}/>);
    expect(markup).toContain('Experimental ReSplat preview');
    expect(markup).toContain('Saved in Recents for review, not quality approved.');
    expect(markup).not.toContain('Not saved in Recents');
  });
  it('labels a VGGT mesh preview without pretending it is a Gaussian splat', () => {
    const url='/api/spaces/room/gaussian/'+'a'.repeat(32);
    const markup=renderToStaticMarkup(<GaussianViewer job={{...job,glb_url:undefined,status:'review_required',gaussian:{
      status:'ready',viewer_url:url+'/viewer',mesh_url:url+'/surface-99.glb',requires_review:true,experimental:true,quality_accepted:false,
    }}} onExit={vi.fn()} onFallback={vi.fn()}/>);
    expect(markup).toContain('Experimental VGGT preview');
    expect(markup).toContain('Not saved in Recents');
    expect(markup).toContain('Download preview mesh');
    expect(markup).not.toContain('Download splat');
    expect(markup).not.toContain('Editable room');
  });
  it('embeds only the local artifact viewer and retains editable and exit controls', () => {
    const markup = renderToStaticMarkup(<GaussianViewer job={job} onExit={vi.fn()} onFallback={vi.fn()} />);
    expect(markup).toContain('title="Photographic room reconstruction"');
    expect(markup).toContain(job.gaussian!.viewer_url);
    expect(markup).toContain('Editable room');
    expect(markup).toContain('Exit room');
  });
  it('does not embed remote or arbitrary URLs', () => {
    for (const viewer_url of ['https://example.com/viewer','//example.com/viewer','/api/spaces/room/model.glb']) {
      expect(renderToStaticMarkup(<GaussianViewer job={{...job,gaussian:{status:'ready',viewer_url}}} onExit={vi.fn()} onFallback={vi.fn()} />)).not.toContain('<iframe');
    }
  });
  it('offers export without a nonexistent editable room for standalone captures', () => {
    const markup = renderToStaticMarkup(<GaussianViewer job={{...job,glb_url:undefined,gaussian:{...job.gaussian!,ply_url:'/api/spaces/room/gaussian/'+'a'.repeat(32)+'/scene.ply'}}} onExit={vi.fn()} onFallback={vi.fn()} />);
    expect(markup).toContain('Download splat');
    expect(markup).not.toContain('Editable room');
    expect(markup).toContain('Exit room');
  });
});
