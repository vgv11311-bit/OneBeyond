import React, { useEffect, useRef, useState } from 'react';
import { Compass, Menu, Sparkles } from 'lucide-react';
import { currentSiteRoute, navigateSite, sitePath, SiteRoute } from '../siteRoutes';

interface NavbarProps {
  onOpenUpload: () => void;
  onSelectSample: () => void;
  onHomeClick: () => void;
  activeView: string;
}

export const Navbar: React.FC<NavbarProps> = ({ onOpenUpload, onSelectSample, onHomeClick, activeView }) => {
  const [siteRoute, setSiteRoute] = useState(currentSiteRoute);
  const [navRevealed, setNavRevealed] = useState(false);
  const headerRef = useRef<HTMLElement>(null);
  const mobileMenuRef = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const update = () => { setSiteRoute(currentSiteRoute()); setNavRevealed(false); };
    window.addEventListener('popstate', update);
    return () => window.removeEventListener('popstate', update);
  }, []);
  useEffect(() => {
    if (!navRevealed || siteRoute === 'home') return;
    const dismissOutside = (event: PointerEvent) => {
      if (!headerRef.current?.contains(event.target as Node)) setNavRevealed(false);
    };
    document.addEventListener('pointerdown', dismissOutside);
    return () => document.removeEventListener('pointerdown', dismissOutside);
  }, [navRevealed, siteRoute]);
  const goTo = (route: SiteRoute) => {
    if (mobileMenuRef.current) mobileMenuRef.current.open = false;
    setNavRevealed(false);
    navigateSite(route);
  };
  const pageLink = (route: SiteRoute, label: string) => <a key={route} href={sitePath(route)} aria-current={siteRoute === route ? 'page' : undefined}
    className={`aethera-nav-link ${siteRoute === route ? 'is-active' : ''}`}
    onClick={(event) => { if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return; event.preventDefault(); goTo(route); }}>{label}</a>;

  if (activeView === 'landing') {
    const onHome = siteRoute === 'home';
    return (
      <header ref={headerRef} className={`aethera-nav aethera-nav-dark inset-x-0 top-0 z-40 px-5 py-5 sm:px-8 sm:py-6${onHome ? ' aethera-nav-home' : ' aethera-nav-subpage'}${navRevealed ? ' is-revealed' : ''}`}
        onMouseEnter={onHome ? undefined : () => { if (window.matchMedia('(hover: hover) and (pointer: fine)').matches) setNavRevealed(true); }}
        onMouseLeave={onHome ? undefined : () => { if (window.matchMedia('(hover: hover) and (pointer: fine)').matches && !mobileMenuRef.current?.open) setNavRevealed(false); }}
        onFocusCapture={onHome ? undefined : () => setNavRevealed(true)}
        onBlurCapture={onHome ? undefined : (event) => { if (!event.currentTarget.contains(event.relatedTarget)) setNavRevealed(false); }}
        onKeyDown={onHome ? undefined : (event) => { if (event.key === 'Escape') { mobileMenuRef.current?.removeAttribute('open'); setNavRevealed(false); } }}>
        {!onHome && <button type="button" className="site-nav-peek" aria-label="Show navigation" aria-expanded={navRevealed} aria-controls="site-main-navigation" onClick={() => setNavRevealed(true)}><Menu size={18} aria-hidden="true" /><span>Menu</span></button>}
        <nav id="site-main-navigation" aria-label="Main navigation" inert={!onHome && !navRevealed} className="site-nav-surface mx-auto flex max-w-7xl items-center justify-between gap-4">
          <button type="button" onClick={() => { goTo('home'); onHomeClick(); }} className="aethera-brand group flex shrink-0 items-center gap-3 rounded-full" aria-label="One Beyond home">
            <span className="aethera-brand-icon relative block h-11 w-11 shrink-0 overflow-hidden rounded-full bg-[var(--ob-orange)]">
              <img src="/one-beyond-logo.png" alt="" className="absolute left-1/2 top-[-10px] h-[83px] w-[83px] max-w-none -translate-x-1/2" />
            </span>
            <span className="font-display text-[1.7rem] leading-none tracking-tight sm:text-3xl">One Beyond</span>
          </button>
          <div className="hidden items-center gap-4 lg:flex xl:gap-6">
            {pageLink('home', 'Home')}
            {pageLink('studio', 'Studio')}
            {pageLink('sample', 'Sample Space')}
            {pageLink('recent', 'Your Spaces')}
            {pageLink('about', 'About')}
            {pageLink('journal', 'Journal')}
            {pageLink('contact', 'Contact')}
          </div>
          <div className="flex items-center gap-2">
            <details ref={mobileMenuRef} className="site-mobile-menu lg:hidden"><summary className="aethera-text-button rounded-full px-3 py-2 text-sm font-medium">Menu</summary><div className="site-mobile-menu-panel">{pageLink('home', 'Home')}{pageLink('studio', 'Studio')}{pageLink('sample', 'Sample Space')}{pageLink('recent', 'Your Spaces')}{pageLink('about', 'About')}{pageLink('journal', 'Journal')}{pageLink('contact', 'Contact')}</div></details>
            <button type="button" onClick={() => goTo('studio')} className="aethera-button primary-button hidden rounded-full px-4 py-2.5 text-xs font-medium sm:inline-flex sm:px-6 sm:text-sm">Open Studio</button>
          </div>
        </nav>
      </header>
    );
  }

  return (
    <header className="fixed inset-x-0 top-0 z-40 px-6 py-4 font-['Inter',sans-serif]">
      <nav aria-label="Main navigation" className="mx-auto flex max-w-7xl items-center justify-between">
        <button type="button" onClick={() => { goTo('home'); onHomeClick(); }} className="aethera-brand flex items-center gap-3 rounded-full text-left text-[var(--ob-paper)] transition-opacity hover:opacity-80" aria-label="One Beyond home">
          <span className="aethera-brand-icon relative block h-9 w-9 shrink-0 overflow-hidden rounded-full bg-[var(--ob-orange)]"><img src="/one-beyond-logo.png" alt="" className="absolute left-1/2 top-[-8px] h-[68px] w-[68px] max-w-none -translate-x-1/2" /></span>
          <span className="font-display text-[1.6rem] leading-none">One Beyond</span>
        </button>
        <div className="flex items-center gap-3">
          <button type="button" onClick={onSelectSample} className="hidden items-center gap-2 rounded-full border border-[#e6d5bd]/25 bg-[#e6d5bd]/10 px-3.5 py-1.5 text-xs font-medium text-[#e6d5bd] transition-colors hover:bg-[#e6d5bd]/20 sm:flex"><Compass className="h-3.5 w-3.5" /> Sample Space</button>
          <button type="button" onClick={onOpenUpload} className="aethera-button rounded-full bg-[var(--ob-orange)] px-4 py-2 text-xs font-semibold text-white"><Sparkles className="h-3.5 w-3.5" /> Create 3D Space</button>
        </div>
      </nav>
    </header>
  );
};
