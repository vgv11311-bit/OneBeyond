import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import Lenis from 'lenis';
import 'lenis/dist/lenis.css';
import { SpaceJob } from '../types';
import { LivingBlueprint } from './LivingBlueprint';
import { SitePages } from './SitePages';
import { currentSiteRoute, navigateSite, sitePath, SiteRoute } from '../siteRoutes';

interface LandingHeroProps {
  onOpenUpload: () => void;
  onExploreSample: () => void;
  onSelectSpace: (space: SpaceJob) => void;
  recentSpaces: SpaceJob[];
  spacesStatus: 'loading' | 'ready' | 'error';
  spacesError: string;
  onRetrySpaces: () => void;
  isUploadOpen: boolean;
}

const FooterLink: React.FC<{ route: SiteRoute; children: React.ReactNode }> = ({ route, children }) => <a href={sitePath(route)} onClick={(event) => {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  navigateSite(route);
}}>{children}</a>;

export const LandingHero: React.FC<LandingHeroProps> = ({ onOpenUpload, onExploreSample, onSelectSpace, recentSpaces, spacesStatus, spacesError, onRetrySpaces, isUploadOpen }) => {
  const [route, setRoute] = useState(currentSiteRoute);
  const lenisRef = useRef<Lenis | null>(null);
  const pageRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const updateRoute = () => setRoute(currentSiteRoute());
    window.addEventListener('popstate', updateRoute);
    return () => window.removeEventListener('popstate', updateRoute);
  }, []);

  useEffect(() => {
    const lenis = new Lenis({
      autoRaf: true,
      lerp: route === 'sample' ? 0.075 : route === 'about' ? 0.085 : 0.1,
      smoothWheel: true,
      anchors: false,
      stopInertiaOnNavigate: true,
      respectReducedMotion: true,
      syncTouch: false,
      prevent: (node) => Boolean(node.closest('[data-lenis-prevent]')),
    });
    lenisRef.current = lenis;
    return () => {
      lenisRef.current = null;
      lenis.destroy();
    };
  }, [route]);

  useEffect(() => {
    if (isUploadOpen) lenisRef.current?.stop();
    else lenisRef.current?.start();
  }, [isUploadOpen, route]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      lenisRef.current?.resize();
      lenisRef.current?.scrollTo(0, { immediate: true, force: true });
      window.scrollTo(0, 0);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [route]);

  useEffect(() => {
    const page = pageRef.current;
    if (!page) return;
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let frame = 0;
    const updateProgress = () => {
      frame = 0;
      const distance = Math.max(1, page.offsetHeight - window.innerHeight);
      page.style.setProperty('--page-progress', String(Math.min(1, Math.max(0, window.scrollY / distance))));
      if (!preference.matches) {
        const about = page.querySelector<HTMLElement>('.about-section');
        if (about) {
          const rect = about.getBoundingClientRect();
          const reveal = Math.min(1, Math.max(0, (window.innerHeight - rect.top) / (window.innerHeight + rect.height)));
          about.style.setProperty('--about-rotate', `${(reveal * 28 - 14).toFixed(2)}deg`);
        }
        page.querySelectorAll<HTMLElement>('[data-scroll-drift]').forEach((node) => {
          const rect = node.getBoundingClientRect();
          const progress = Math.min(1, Math.max(0, (window.innerHeight - rect.top) / (window.innerHeight + rect.height)));
          const kind = node.dataset.scrollDrift;
          if (kind === 'studio') node.style.transform = `translate3d(0, ${((.5 - progress) * 64).toFixed(1)}px, 0)`;
          if (kind === 'sample') node.style.transform = `translate3d(0, ${((.5 - progress) * 46).toFixed(1)}px, 0) rotate(${((progress - .5) * 8).toFixed(1)}deg)`;
          if (kind === 'about') node.style.transform = `translate3d(0, ${((.5 - progress) * 28).toFixed(1)}px, 0) rotate(${((progress - .5) * 12).toFixed(1)}deg)`;
        });
        page.querySelectorAll<HTMLElement>('[data-scroll-motion]').forEach((node) => {
          const rect = node.getBoundingClientRect();
          const progress = Math.min(1, Math.max(0, (window.innerHeight - rect.top) / (window.innerHeight + rect.height)));
          switch (node.dataset.scrollMotion) {
            case 'studio-photo': node.style.transform = `scale(${(1.09 - progress * .08).toFixed(3)})`; break;
            case 'sample-photo': node.style.transform = `scale(${(1.07 - progress * .05).toFixed(3)}) translate3d(${((progress - .5) * 3).toFixed(2)}%, 0, 0)`; break;
            case 'sample-immersion': {
              const section = node.closest<HTMLElement>('.sample-immersion-scroll');
              const travel = section ? Math.max(1, section.offsetHeight - window.innerHeight) : 1;
              const sectionProgress = section ? Math.min(1, Math.max(0, -section.getBoundingClientRect().top / travel)) : progress;
              node.style.transform = `scale(${(1.23 - sectionProgress * .22).toFixed(3)})`;
              break;
            }
            case 'recent-count': node.style.transform = `translate3d(0, ${((.5 - progress) * 90).toFixed(1)}px, 0)`; break;
            case 'journal-title': node.style.transform = `translate3d(${((.5 - progress) * 42).toFixed(1)}px, 0, 0)`; break;
            case 'contact-type': node.style.transform = `translate3d(${((progress - .5) * 190).toFixed(1)}px, 0, 0)`; break;
          }
        });
      }
    };
    const requestProgress = () => { if (!frame) frame = window.requestAnimationFrame(updateProgress); };
    const targets = Array.from(page.querySelectorAll<HTMLElement>('[data-scroll-reveal]'));
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -4% 0px' });
    if (!preference.matches) {
      targets.forEach((target) => observer.observe(target));
      page.classList.add('scroll-enhanced');
    }
    window.addEventListener('scroll', requestProgress, { passive: true });
    window.addEventListener('resize', requestProgress);
    requestProgress();
    return () => {
      observer.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener('scroll', requestProgress);
      window.removeEventListener('resize', requestProgress);
      page.classList.remove('scroll-enhanced');
    };
  }, [recentSpaces.length, route]);

  return (
    <main ref={pageRef} className="aethera-page">
      <div className="site-progress" aria-hidden="true"><span /></div>
      {route === 'home' ? <>
      <LivingBlueprint onOpenUpload={onOpenUpload} />

      <section className="home-proof" aria-labelledby="proof-title">
        <div className="subpage-container home-proof-grid">
          <div className="home-proof-copy" data-scroll-reveal>
            <p className="section-kicker">01 / THE VIEWER</p>
            <h2 id="proof-title" className="font-display">The real test is <em>what you can inspect.</em></h2>
            <p>A room is more useful when you can move through it, question its scale, and look beyond the camera angle. Try the sample viewer before bringing your own space.</p>
            <button type="button" onClick={onExploreSample} className="aethera-button primary-button rounded-full px-7 py-3.5 text-sm font-medium">Enter Sample Viewer <ArrowRight size={16} aria-hidden="true" /></button>
          </div>
          <figure className="home-proof-visual" data-scroll-reveal>
            <img src="/sample-interior-illustrative.png" alt="Illustrative interior scene representing a room to explore" loading="lazy" decoding="async" />
            <figcaption>ILLUSTRATIVE EDITORIAL IMAGE — OPEN THE VIEWER FOR THE ACTUAL SAMPLE</figcaption>
          </figure>
        </div>
      </section>

      <section className="home-evidence" aria-labelledby="evidence-title">
        <div className="subpage-container">
          <p className="section-kicker" data-scroll-reveal>02 / HONEST RECONSTRUCTION</p>
          <h2 id="evidence-title" className="font-display" data-scroll-reveal>See the space. <em>Know the limits.</em></h2>
          <div className="home-evidence-grid">
            <article data-scroll-reveal><span>01 / VISIBLE</span><h3>Start with evidence.</h3><p>Photographs give the room its visible surfaces, objects and relationships.</p></article>
            <article data-scroll-reveal><span>02 / ESTIMATED</span><h3>Keep scale honest.</h3><p>Without a measured width, dimensions remain estimates—not a survey.</p></article>
            <article data-scroll-reveal><span>03 / EXPLORE</span><h3>Inspect the result.</h3><p>The viewer lets you revisit the scene from perspectives the photos cannot show.</p></article>
          </div>
        </div>
      </section>

      <section className="home-discovery" aria-labelledby="discovery-title">
        <div className="home-discovery-orbit" aria-hidden="true"><span /></div>
        <div className="subpage-container home-discovery-inner">
          <p className="section-kicker" data-scroll-reveal>03 / YOUR SPACE</p>
          <h2 id="discovery-title" className="font-display" data-scroll-reveal>Bring a place <em>into focus.</em></h2>
          <p data-scroll-reveal>Have overlapping photographs of a real indoor space? Open the studio to understand the capture requirements, or create a space now.</p>
          <div className="home-discovery-actions" data-scroll-reveal>
            <button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Create 3D Space <ArrowRight aria-hidden="true" size={16} /></button>
            <button type="button" onClick={() => navigateSite('studio')} className="aethera-text-button rounded-full px-5 py-3.5 text-sm font-medium">Open Studio <ArrowRight aria-hidden="true" size={16} /></button>
          </div>
        </div>
      </section>
      </> : <SitePages route={route} onOpenUpload={onOpenUpload} onExploreSample={onExploreSample} onSelectSpace={onSelectSpace} recentSpaces={recentSpaces} spacesStatus={spacesStatus} spacesError={spacesError} onRetrySpaces={onRetrySpaces} />}
      <footer className="site-footer">
        <div className="subpage-container site-footer-grid">
          <div><p className="site-footer-brand font-display">One Beyond</p><p>Real spaces. More ways to see them.</p><small>Independently built by Vihaan Vakharia.</small></div>
          <nav aria-label="Footer navigation"><FooterLink route="studio">Studio</FooterLink><FooterLink route="sample">Sample Space</FooterLink><FooterLink route="recent">Your Spaces</FooterLink><FooterLink route="about">About</FooterLink><FooterLink route="contact">Contact</FooterLink></nav>
        </div>
      </footer>
    </main>
  );
};
