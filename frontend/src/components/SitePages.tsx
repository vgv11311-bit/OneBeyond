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

export const SitePages: React.FC<SitePagesProps> = ({ route, onOpenUpload, onExploreSample, onSelectSpace, recentSpaces, spacesStatus, spacesError, onRetrySpaces }) => {
  const [visibleCount, setVisibleCount] = useState(12);
  useEffect(() => setVisibleCount(12), [route]);
  const visibleSpaces = recentSpaces.slice(0, visibleCount);

  if (route === 'studio') return <div className="subpage studio-page">
    <section className="subpage-hero studio-hero">
      <div className="subpage-container studio-hero-grid">
        <div data-scroll-reveal>
          <p className="section-kicker">THE STUDIO / 01</p>
          <h1 className="font-display">From a frame <em>to a place.</em></h1>
          <p className="subpage-lede">Begin with real room photographs. Follow the reconstruction, then step into an explorable space.</p>
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
        <article data-scroll-reveal><span>01</span><Camera aria-hidden="true" /><div><h3>Capture</h3><p>Bring overlapping views of one indoor space. A measured width helps scale the result; without one, dimensions remain estimates.</p></div></article>
        <article data-scroll-reveal><span>02</span><ScanLine aria-hidden="true" /><div><h3>Reconstruct</h3><p>Watch the visible layout, objects and geometry take shape. The pipeline reports progress and any uncertainty.</p></div></article>
        <article data-scroll-reveal><span>03</span><Compass aria-hidden="true" /><div><h3>Explore</h3><p>Enter the 3D viewer to inspect the result from more than the original camera angle.</p></div></article>
      </div>
    </section>
    <PageEnd title="Ready to see beyond the image?" onOpenUpload={onOpenUpload} />
  </div>;

  if (route === 'sample') return <div className="subpage sample-page">
    <section className="subpage-hero sample-hero">
      <div className="subpage-container sample-hero-grid">
        <div data-scroll-reveal><p className="section-kicker">SAMPLE SPACE / 01</p><h1 className="font-display">A space you can <em>step into.</em></h1><p className="subpage-lede">Try the viewer before uploading your own photos. Walk, look around, and explore how a spatial room feels.</p><button type="button" onClick={onExploreSample} className="aethera-button primary-button rounded-full px-8 py-4 text-sm font-medium">Enter Sample Viewer <ArrowRight size={16} aria-hidden="true" /></button></div>
        <figure className="editorial-visual sample-editorial">
          <div className="editorial-visual-frame"><img src="/sample-interior-illustrative.png" alt="Illustrative light-filled living space viewed from within" data-scroll-motion="sample-photo" /></div>
          <figcaption>ILLUSTRATIVE INTERIOR / EXPLORE THE ACTUAL DEMO BELOW</figcaption>
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
        <div data-scroll-reveal><p>Move through a room, change your view, and see the reconstruction as a spatial scene. It is a demonstration of the viewer, not a promise of exact geometry from every photo.</p><button type="button" onClick={onExploreSample} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Enter Sample Viewer <ArrowRight size={16} aria-hidden="true" /></button></div>
      </div>
    </section>
    <PageEnd title="Now imagine your own space." onOpenUpload={onOpenUpload} />
  </div>;

  if (route === 'recent') return <div className="subpage recent-page">
    <section className="subpage-hero recent-hero"><div className="subpage-container recent-hero-content"><div data-scroll-reveal><p className="section-kicker">YOUR SPACES / 01</p><h1 className="font-display">Places worth <em>revisiting.</em></h1><p className="subpage-lede">Open a saved reconstruction and continue exploring it.</p></div><span className="recent-hero-count font-display" data-scroll-motion="recent-count" aria-hidden="true">{recentSpaces.length ? recentSpaces.length.toString().padStart(2, '0') : spacesStatus === 'ready' ? '00' : '—'}</span></div></section>
    <section className="recent-gallery subpage-container" aria-label="Saved spaces">
      <div className="recent-gallery-heading"><p className="section-kicker">THE COLLECTION</p><p>{recentSpaces.length ? `Showing ${visibleSpaces.length} of ${recentSpaces.length} saved spaces` : spacesStatus === 'ready' ? '0 saved spaces' : 'Saved spaces'}</p></div>
      {spacesStatus === 'loading' && <p className="recent-collection-notice" role="status">{recentSpaces.length ? 'Refreshing saved spaces…' : 'Loading saved spaces…'}</p>}
      {spacesStatus === 'error' && <div className="recent-collection-notice recent-collection-error" role="alert"><span>Could not refresh saved spaces. {recentSpaces.length ? 'Showing the last loaded collection.' : 'No collection has loaded yet.'} {spacesError}</span><button type="button" onClick={onRetrySpaces} className="aethera-button primary-button rounded-full px-5 py-2 text-sm font-medium">Retry</button></div>}
      {recentSpaces.length ? <><div className="recent-gallery-grid">{visibleSpaces.map((space, index) => <button type="button" key={space.job_id} data-testid={`space-${space.job_id}`} onClick={() => onSelectSpace(space)} className="recent-gallery-card aethera-space-card" data-scroll-reveal><span className="recent-gallery-index">{String(index + 1).padStart(2, '0')}</span><Box size={36} aria-hidden="true" /><span className="recent-gallery-card-bottom"><span><strong>{space.name || `Space ${space.job_id}`}</strong><SavedSpaceReview space={space} /><small>{space.metrics && space.metrics.estimated_room_width > 0 && space.metrics.estimated_room_length > 0 ? `${space.metrics.estimated_room_width}m × ${space.metrics.estimated_room_length}m` : space.gaussian?.status === 'ready' ? 'Photographic capture · scale unmeasured' : 'Explorable room'}</small></span><ArrowRight size={18} aria-hidden="true" /></span></button>)}</div>{visibleCount < recentSpaces.length && <button type="button" onClick={() => setVisibleCount(count => count + 12)} className="aethera-button primary-button recent-load rounded-full px-7 py-3.5 text-sm font-medium">Show more spaces <ArrowRight size={16} aria-hidden="true" /></button>}</> : spacesStatus === 'ready' ? <div className="recent-empty" data-scroll-reveal><Layers3 size={42} aria-hidden="true" /><h2 className="font-display">Your collection begins here.</h2><p>Create your first space to see it in this gallery.</p><button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Create 3D Space <ArrowRight size={16} aria-hidden="true" /></button></div> : null}
    </section>
  </div>;

  if (route === 'about') return <div className="subpage about-page">
    <section className="subpage-hero about-hero"><div className="subpage-container about-hero-grid"><div data-scroll-reveal><p className="section-kicker">ABOUT / 01</p><h1 className="font-display">There is more to a place <em>than one view.</em></h1></div><div className="about-brand-mark" data-scroll-drift="about" aria-hidden="true"><img src="/one-beyond-logo.png" alt="" /></div></div></section>
    <section className="about-manifesto subpage-container"><p className="section-kicker" data-scroll-reveal>THE IDEA</p><h2 className="font-display" data-scroll-reveal>Make the physical world <em>more explorable.</em></h2><p data-scroll-reveal>One Beyond is an independent spatial computing project turning photographs of real indoor spaces into walkable 3D scenes. The experience emphasizes clear evidence, honest estimates, and the freedom to look closer.</p></section>
    <section className="creator-section" aria-labelledby="creator-title"><div className="subpage-container creator-grid"><div className="creator-editorial" aria-hidden="true"><span>INDEPENDENT BY DESIGN</span><strong className="font-display">One person.<br /><em>A wider view.</em></strong><i /></div><div className="creator-copy" data-scroll-reveal><p className="section-kicker">THE PERSON BEHIND ONE BEYOND</p><h2 id="creator-title" className="font-display">Built by <em>Vihaan Vakharia.</em></h2><p>Vihaan is a <strong>15-year-old Grade 11 student</strong> building One Beyond around an ambitious idea: a real place should be something you can step into, not just look at. He is taking on spatial reconstruction independently—working through the difficult parts of geometry, evidence, and experience to make that idea tangible.</p></div></div></section>
    <section className="about-values"><div className="subpage-container"><p className="section-kicker" data-scroll-reveal>DESIGN PRINCIPLES</p><div><article data-scroll-reveal><span>01</span><h3>Real spaces first.</h3><p>Every result begins with the user's room, not a prebuilt room template.</p></article><article data-scroll-reveal><span>02</span><h3>Clarity over claims.</h3><p>Estimated scale and unverified details should be visible, not disguised as certainty.</p></article><article data-scroll-reveal><span>03</span><h3>Built to explore.</h3><p>The viewer makes a reconstruction inspectable from more than one angle.</p></article></div></div></section>
    <PageEnd title="See what a room can become." onOpenUpload={onOpenUpload} />
  </div>;

  if (route === 'journal') return <div className="subpage journal-page">
    <section className="subpage-hero journal-hero"><div className="subpage-container" data-scroll-reveal><p className="section-kicker">JOURNAL / 01</p><h1 className="font-display">Notes from the <em>edge of the image.</em></h1><p className="subpage-lede">Observations from building an experience for real spaces.</p></div></section>
    <section className="journal-entries subpage-container" aria-label="Build notes"><article data-scroll-reveal><span>01 / THE CHALLENGE</span><h2 className="font-display" data-scroll-motion="journal-title">A photograph is not a floor plan.</h2><p>Depth and hidden surfaces must be inferred. That makes evidence, scale, and clear uncertainty important parts of the experience.</p></article><article data-scroll-reveal><span>02 / THE APPROACH</span><h2 className="font-display" data-scroll-motion="journal-title">Reconstruct, then inspect.</h2><p>A walkable scene lets you find both what worked and what still needs refinement from angles the photo never showed.</p></article><article data-scroll-reveal><span>03 / WHAT'S NEXT</span><h2 className="font-display" data-scroll-motion="journal-title">Better with every space.</h2><p>The focus is improving layout, object placement, and visual fidelity without hiding when the source photos leave information unknown.</p></article></section>
    <PageEnd title="Explore the idea in your own space." onOpenUpload={onOpenUpload} />
  </div>;

  return <div className="subpage contact-page">
    <section className="subpage-hero contact-hero"><div className="subpage-container contact-hero-content" data-scroll-reveal><p className="section-kicker">CONTACT / 01</p><h1 className="font-display">A conversation <em>starts here.</em></h1><p className="subpage-lede">One Beyond is an independent project by Vihaan Vakharia. A verified public email or social contact has not yet been provided for this site.</p><div className="contact-status"><span>CONTACT CHANNEL</span><strong>Awaiting verified details</strong><p>No address or profile is shown until Vihaan confirms where visitors should write.</p></div><div className="contact-actions"><button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-8 py-4 text-sm font-medium">Create 3D Space <ArrowRight size={16} aria-hidden="true" /></button><button type="button" onClick={onExploreSample} className="aethera-button contact-secondary rounded-full px-8 py-4 text-sm font-medium">Enter Sample Viewer <Compass size={16} aria-hidden="true" /></button></div></div><span className="contact-hero-word font-display" data-scroll-motion="contact-type" aria-hidden="true">Connect</span></section>
    <section className="contact-after subpage-container"><div data-scroll-reveal><Sparkles aria-hidden="true" /><p className="section-kicker">THE CREATOR</p><h2 className="font-display">Built by <em>Vihaan Vakharia.</em></h2><PageLink route="about" className="contact-about-link">About the creator <ArrowRight size={16} aria-hidden="true" /></PageLink></div></section>
  </div>;
};
