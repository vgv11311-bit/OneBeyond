import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { UploadModal } from './UploadModal';

describe('sparse-view capture guidance', () => {
  it('lists ReSplat as an explicit experimental COLMAP option', () => {
    const html=renderToStaticMarkup(<UploadModal isOpen onClose={vi.fn()} onSubmit={vi.fn()} isSubmitting={false} engines={[
      {id:'resplat_experimental',name:'COLMAP + ReSplat — Experimental',description:'Native Gaussian refinement; CUDA required',version:'0.1.0',is_default:false,experimental:true,can_run:false},
    ]}/>);
    expect(html).toContain('COLMAP + ReSplat');expect(html).toContain('Runtime needs attention');
  });
  it('advertises VGGT as experimental and reports unavailable runtime without assuming CUDA-only', () => {
    const html=renderToStaticMarkup(<UploadModal isOpen onClose={vi.fn()} onSubmit={vi.fn()} isSubmitting={false} engines={[
      {id:'vggt_experimental',name:'VGGT joint geometry — Experimental',description:'Joint camera/depth preview',version:'0.1.0',is_default:false,experimental:true,can_run:false},
    ]}/>);
    expect(html).toContain('VGGT joint geometry');
    expect(html).toContain('Runtime needs attention');
    expect(html).toContain('aria-pressed="false"');
  });
  it('explains that installed local VGGT can be blocked by RAM rather than a missing download', () => {
    const html=renderToStaticMarkup(<UploadModal isOpen onClose={vi.fn()} onSubmit={vi.fn()} isSubmitting={false} engines={[
      {id:'vggt_experimental',name:'VGGT joint geometry — Experimental',description:'Local CPU',version:'0.2.0',is_default:false,experimental:true,can_run:false,availability_code:'vggt_local_memory_low'},
    ]}/>);
    expect(html).toContain('Installed locally');expect(html).toContain('free RAM needed');
  });
  it('recommends 5–8 photos without making more photos mandatory or promising unseen accuracy', () => {
    const html = renderToStaticMarkup(<UploadModal isOpen onClose={vi.fn()} onSubmit={vi.fn()} engines={[]} isSubmitting={false} />);
    expect(html).toContain('Add 5–8 overlapping photos');
    expect(html).toContain('up to 10');
    expect(html).toContain('More are optional (3–60 accepted)');
    expect(html).toContain('Unseen surfaces remain uncertain');
    expect(html).not.toContain('20–40');
  });
});
