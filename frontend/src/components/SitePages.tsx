import React, { useEffect, useState } from 'react';
import { ArrowRight, Box, Camera, Compass, Layers3, ScanLine, Sparkles } from 'lucide-react';
import { SpaceJob } from '../types';
import { SavedSpaceReview } from './SavedSpaceReview';
import { navigateSite, sitePath, SiteRoute } from '../siteRoutes';

interface SitePagesProps {
  route: Exclude<SiteRoute, 'home'>;
  onOpenUpload: () => void;
  onExploreSample: () => void;
  onSelectSpace: (space: SpaceJob) => void;
  recentSpaces: SpaceJob[];
  spacesStatus: 'loading' | 'ready' | 'error';
  spacesError: string;
  onRetrySpaces: () => void;
}

const PageLink: React.FC<{ route: SiteRoute; children: React.ReactNode; className?: string }> = ({ route, children, className }) => (
  <a href={sitePath(route)} className={className} onClick={(event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    navigateSite(route);
  }}>{children}</a>
);

const PageEnd: React.FC<{ title: string; onOpenUpload: () => void }> = ({ title, onOpenUpload }) => (
  <section className="subpage-end">
    <div className="subpage-container subpage-end-inner" data-scroll-reveal>
      <p className="section-kicker">ONE BEYOND / NEXT STEP</p>
      <h2 className="font-display">{title}</h2>
      <button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Create 3D Space <ArrowRight size={16} aria-hidden="true" /></button>
    </div>
  </section>
);

export const SitePages: React.FC<SitePagesProps> = ({ route, onOpenUpload, onExploreSample, onSelectSpace, recentSpaces, spacesStatus, onRetrySpaces }) => {
  const [visibleCount, setVisibleCount] = useState(12);
  useEffect(() => setVisibleCount(12), [route]);
  const visibleSpaces = recentSpaces.slice(0, visibleCount);

  if (route === 'studio') return <div className="subpage studio-page">
    <section className="subpage-hero studio-hero">
      <div className="subpage-container studio-hero-grid">
        <div data-scroll-reveal>
          <p className="section-kicker">THE STUDIO / 01</p>
          <h1 className="font-display">From room captures <em>to an explorable space.</em></h1>
          <p className="subpage-lede">Start with overlapping room photos or a walkthrough video. Follow the reconstruction progress, then explore the result.</p>
          <button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-8 py-4 text-sm font-medium">Create 3D Space <ArrowRight size={16} aria-hidden="true" /></button>
        </div>
        <figure className="editorial-visual studio-editorial">
          <div className="editorial-visual-frame"><img src="/studio-interior-illustrative.png" alt="Illustrative warm interior seen through a doorway" data-scroll-motion="studio-photo" /></div>
          <figcaption>AN ILLUSTRATIVE SPACE / NOT A RECONSTRUCTION RESULT</figcaption>
        </figure>
      </div>
    </section>
    <section className="studio-sequence subpage-container" aria-labelledby="studio-sequence-title">
      <div className="studio-sequence-head"><p className="section-kicker">A SIMPLE PATH</p><h2 id="studio-sequence-title" className="font-display">Three moments. <em>One new perspective.</em></h2></div>
      <div className="studio-sequence-list">
        <article data-scroll-reveal><span>01</span><Camera aria-hidden="true" /><div><h3>Capture</h3><p>Capture overlapping views of one indoor space. Dimensions are approximate unless independently verified.</p></div></article>
        <article data-scroll-reveal><span>02</span><ScanLine aria-hidden="true" /><div><h3>Reconstruct</h3><p>Follow reconstruction progress and review the available quality notices.</p></div></article>
        <article data-scroll-reveal><span>03</span><Compass aria-hidden="true" /><div><h3>Explore</h3><p>Enter the 3D viewer to inspect the result from more than the original camera angle.</p></div></article>
      </div>
    </section>
    <PageEnd title="Ready to see beyond the image?" onOpenUpload={onOpenUpload} />
  </div>;

  if (route === 'sample') return <div className="subpage sample-page">
    <section className="subpage-hero sample-hero">
      <div className="subpage-container sample-hero-grid">
        <div data-scroll-reveal><p className="section-kicker">SAMPLE SPACE / 01</p><h1 className="font-display">A space you can <em>step into.</em></h1><p className="subpage-lede">Try the viewer before uploading your own room captures. Move around and explore the room from different viewpoints.</p><button type="button" onClick={onExploreSample} className="aethera-button primary-button rounded-full px-8 py-4 text-sm font-medium">Open sample <ArrowRight size={16} aria-hidden="true" /></button></div>
        <figure className="editorial-visual sample-editorial">
          <div className="editorial-visual-frame"><img src="/sample-interior-illustrative.png" alt="Illustrative light-filled living space viewed from within" data-scroll-motion="sample-photo" /></div>
          <figcaption>Illustrative image. Select Open sample to explore the demo.</figcaption>
        </figure>
      </div>
    </section>
    <section className="sample-immersion-scroll" aria-label="A change of perspective">
      <div className="sample-immersion-sticky">
        <img src="/sample-interior-illustrative.png" alt="" data-scroll-motion="sample-immersion" />
        <div className="sample-immersion-copy"><span>MOVE BEYOND THE FRAME</span><strong className="font-display">Look closer.<br /><em>Then step inside.</em></strong></div>
      </div>
    </section>
    <section className="sample-drift subpage-container">
      <p className="section-kicker" data-scroll-reveal>LOOK BEYOND ONE VIEW</p>
      <div className="sample-drift-grid">
        <h2 className="font-display" data-scroll-reveal>Not a still image. <em>A point of view you control.</em></h2>
        <div data-scroll-reveal><p>Move through the sample room and explore it from different viewpoints. This sample demonstrates the viewer. Reconstruction quality varies with capture coverage, and geometry may contain inaccuracies.</p><button type="button" onClick={onExploreSample} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Open sample <ArrowRight size={16} aria-hidden="true" /></button></div>
      </div>
    </section>
    <PageEnd title="Now imagine your own space." onOpenUpload={onOpenUpload} />
  </div>;

  if (route === 'recent') return <div className="subpage recent-page">
    <section className="subpage-hero recent-hero"><div className="subpage-container recent-hero-content"><div data-scroll-reveal><p className="section-kicker">YOUR SPACES / 01</p><h1 className="font-display">Places worth <em>revisiting.</em></h1><p className="subpage-lede">Open a saved reconstruction and continue exploring it.</p></div><span className="recent-hero-count font-display" data-scroll-motion="recent-count" aria-hidden="true">{recentSpaces.length ? recentSpaces.length.toString().padStart(2, '0') : spacesStatus === 'ready' ? '00' : '—'}</span></div></section>
    <section className="recent-gallery subpage-container" aria-label="Saved spaces">
      <div className="recent-gallery-heading"><p className="section-kicker">THE COLLECTION</p><p>{recentSpaces.length ? `Showing ${visibleSpaces.length} of ${recentSpaces.length} saved spaces` : spacesStatus === 'ready' ? '0 saved spaces' : 'Saved spaces'}</p></div>
      {spacesStatus === 'loading' && <p className="recent-collection-notice" role="status">{recentSpaces.length ? 'Refreshing saved spaces…' : 'Loading saved spaces…'}</p>}
      {spacesStatus === 'error' && <div className="recent-collection-notice recent-collection-error" role="alert"><span>Saved spaces are temporarily unavailable. Please try again.{recentSpaces.length ? ' Showing the last loaded collection.' : ''}</span><button type="button" onClick={onRetrySpaces} className="aethera-button primary-button rounded-full px-5 py-2 text-sm font-medium">Retry</button></div>}
      {recentSpaces.length ? <><div className="recent-gallery-grid">{visibleSpaces.map((space, index) => <button type="button" key={space.job_id} data-testid={`space-${space.job_id}`} onClick={() => onSelectSpace(space)} className="recent-gallery-card aethera-space-card" data-scroll-reveal><span className="recent-gallery-index">{String(index + 1).padStart(2, '0')}</span><Box size={36} aria-hidden="true" /><span className="recent-gallery-card-bottom"><span><strong>{space.name || `Space ${space.job_id}`}</strong><SavedSpaceReview space={space} /><small>{space.metrics && space.metrics.estimated_room_width > 0 && space.metrics.estimated_room_length > 0 ? `${space.metrics.estimated_room_width}m × ${space.metrics.estimated_room_length}m` : space.gaussian?.status === 'ready' ? 'Photographic capture · scale unmeasured' : 'Explorable room'}</small></span><ArrowRight size={18} aria-hidden="true" /></span></button>)}</div>{visibleCount < recentSpaces.length && <button type="button" onClick={() => setVisibleCount(count => count + 12)} className="aethera-button primary-button recent-load rounded-full px-7 py-3.5 text-sm font-medium">Show more spaces <ArrowRight size={16} aria-hidden="true" /></button>}</> : spacesStatus === 'ready' ? <div className="recent-empty" data-scroll-reveal><Layers3 size={42} aria-hidden="true" /><h2 className="font-display">Your collection begins here.</h2><p>Create your first space to see it in this gallery.</p><button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Create 3D Space <ArrowRight size={16} aria-hidden="true" /></button></div> : null}
    </section>
  </div>;

  if (route === 'about') return <div className="subpage about-page">
    <section className="subpage-hero about-hero"><div className="subpage-container about-hero-grid"><div data-scroll-reveal><p className="section-kicker">ABOUT / 01</p><h1 className="font-display">There is more to a place <em>than one view.</em></h1></div><div className="about-brand-mark" data-scroll-drift="about" aria-hidden="true"><img src="/one-beyond-logo.png" alt="" /></div></div></section>
    <section className="about-manifesto subpage-container"><p className="section-kicker" data-scroll-reveal>THE IDEA</p><h2 className="font-display" data-scroll-reveal>Make the physical world <em>more explorable.</em></h2><p data-scroll-reveal>I built One Beyond to turn room photos or walkthrough videos into explorable 3D scenes. Reconstructed details and dimensions may be approximate.</p></section>
    <section className="creator-section" aria-labelledby="creator-title"><div className="subpage-container creator-grid"><div className="creator-editorial" aria-hidden="true"><span>INDEPENDENT BY DESIGN</span><strong className="font-display">My idea.<br /><em>A wider view.</em></strong><i /></div><div className="creator-copy" data-scroll-reveal><p className="section-kicker">MY STORY</p><h2 id="creator-title" className="font-display">I'm <em>Vihaan Vakharia.</em></h2><p>I'm a <strong>15-year-old Grade 11 student</strong>, and I'm building One Beyond around an ambitious idea: you should be able to explore a real place, not just look at a photo of it. I'm developing the project independently, bringing spatial reconstruction and interface design together to make real spaces more explorable.</p></div></div></section>
    <section className="about-values"><div className="subpage-container"><p className="section-kicker" data-scroll-reveal>DESIGN PRINCIPLES</p><div><article data-scroll-reveal><span>01</span><h3>Real spaces first.</h3><p>User-created reconstructions begin with your room captures.</p></article><article data-scroll-reveal><span>02</span><h3>Clarity over claims.</h3><p>Check the available quality notices before relying on dimensions or reconstructed details.</p></article><article data-scroll-reveal><span>03</span><h3>Built to explore.</h3><p>The viewer makes a reconstruction inspectable from more than one angle.</p></article></div></div></section>
    <PageEnd title="See what a room can become." onOpenUpload={onOpenUpload} />
  </div>;

  if (route === 'journal') return <div className="subpage journal-page">
    <section className="subpage-hero journal-hero"><div className="subpage-container" data-scroll-reveal><p className="section-kicker">JOURNAL / 01</p><h1 className="font-display">Behind <em>the build.</em></h1><p className="subpage-lede">Observations from building an experience for real spaces.</p></div></section>
    <section className="journal-entries subpage-container" aria-label="Build notes"><article data-scroll-reveal><span>01 / THE CHALLENGE</span><h2 className="font-display" data-scroll-motion="journal-title">A photograph is not a floor plan.</h2><p>Depth is estimated from the available views. Surfaces not captured may remain incomplete or uncertain. Review the result with those limits in mind.</p></article><article data-scroll-reveal><span>02 / THE APPROACH</span><h2 className="font-display" data-scroll-motion="journal-title">Reconstruct, then inspect.</h2><p>A walkable scene lets you find both what worked and what still needs refinement from angles the photo never showed.</p></article><article data-scroll-reveal><span>03 / WHAT'S NEXT</span><h2 className="font-display" data-scroll-motion="journal-title">What comes next.</h2><p>The focus is improving layout, object placement, and visual fidelity without hiding when the source photos leave information unknown.</p></article></section>
    <PageEnd title="Explore the idea in your own space." onOpenUpload={onOpenUpload} />
  </div>;

  return <div className="subpage contact-page">
    <section className="subpage-hero contact-hero"><div className="subpage-container contact-hero-content" data-scroll-reveal><p className="section-kicker">ABOUT THE PROJECT</p><h1 className="font-display">About <em>One Beyond.</em></h1><p className="subpage-lede">I'm Vihaan Vakharia, and I built One Beyond independently.</p><div className="contact-actions"><button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-8 py-4 text-sm font-medium">Create 3D Space <ArrowRight size={16} aria-hidden="true" /></button><PageLink route="about" className="aethera-button contact-secondary rounded-full px-8 py-4 text-sm font-medium">Read my story <ArrowRight size={16} aria-hidden="true" /></PageLink></div></div><span className="contact-hero-word font-display" data-scroll-motion="contact-type" aria-hidden="true">Explore</span></section>
    <section className="contact-after subpage-container"><div data-scroll-reveal><Sparkles aria-hidden="true" /><p className="section-kicker">MY STORY</p><h2 className="font-display">I'm <em>Vihaan Vakharia.</em></h2><PageLink route="about" className="contact-about-link">Read my story <ArrowRight size={16} aria-hidden="true" /></PageLink></div></section>
  </div>;
};
