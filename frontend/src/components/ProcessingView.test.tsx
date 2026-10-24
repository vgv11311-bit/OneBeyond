import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { ProcessingView } from './ProcessingView';
import { SpaceJob } from '../types';

describe('experimental VGGT review state',()=>{
  it('offers inspection without announcing completion or publication',()=>{
    const job:SpaceJob={job_id:'experimental',name:'Test fixture',created_at:'2026-10-22',input_type:'photos',files_count:5,stage:'Review required',status:'review_required',engine_id:'vggt_experimental',progress:100,logs:[],gaussian:{status:'ready',requires_review:true,quality_accepted:false}};
    const html=renderToStaticMarkup(<ProcessingView job={job} onEnterViewer={vi.fn()} onCancel={vi.fn()} onRetry={vi.fn()} isRetrying={false}/>);
    expect(html).toContain('VGGT Experimental Preview — Review Required');
    expect(html).toContain('Inspect Experimental Preview');
    expect(html).toContain('not in Recents');
    expect(html).toContain('VGGT Camera + Depth');
    expect(html).not.toContain('3D Spatial Room Ready');
    expect(html).not.toContain('Enter Immersive 3D Space');
    expect(html).not.toContain('Publish Walkthrough');
  });
});
