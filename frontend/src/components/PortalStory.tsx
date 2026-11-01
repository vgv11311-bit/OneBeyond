import React, { useEffect, useRef } from 'react';

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const range = (value: number, start: number, end: number) => clamp((value - start) / (end - start));

/** A decorative, scroll-linked explanation of photo-to-space reconstruction. */
export const PortalStory: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const orbitRef = useRef<HTMLDivElement>(null);
  const roomRef = useRef<HTMLDivElement>(null);
  const roomFillRef = useRef<SVGGElement>(null);
  const copyRefs = useRef<(HTMLDivElement | null)[]>([]);
  const progressRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');

    let frame = 0;
    const update = () => {
      frame = 0;
      if (preference.matches) return;
      const section = sectionRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const travel = Math.max(1, rect.height - window.innerHeight);
      const progress = clamp(-rect.top / travel);
      const orbitExit = range(progress, 0.12, 0.38);
      const roomEnter = range(progress, 0.17, 0.54);
      const roomSettle = range(progress, 0.48, 0.88);

      if (orbitRef.current) {
        orbitRef.current.style.opacity = String(1 - orbitExit);
        orbitRef.current.style.transform = `translate3d(0, ${-36 * orbitExit}px, 0) rotate(${75 * orbitExit}deg) scale(${1 + 1.75 * orbitExit})`;
      }
      if (roomRef.current) {
        roomRef.current.style.opacity = String(roomEnter);
        roomRef.current.style.transform = `translate3d(0, ${48 * (1 - roomEnter) - 18 * roomSettle}px, 0) scale(${0.74 + 0.34 * roomEnter + 0.08 * roomSettle})`;
      }
      if (roomFillRef.current) roomFillRef.current.style.opacity = String(range(progress, 0.48, 0.82));

      const visibility = [
        1 - range(progress, 0.08, 0.28),
        range(progress, 0.29, 0.43) * (1 - range(progress, 0.60, 0.72)),
        range(progress, 0.73, 0.89),
      ];
      copyRefs.current.forEach((node, index) => {
        if (!node) return;
        node.style.opacity = String(visibility[index]);
        node.style.transform = `translate3d(0, ${24 * (1 - visibility[index])}px, 0)`;
      });
      if (progressRef.current) progressRef.current.style.transform = `scaleX(${progress})`;
    };
    const requestUpdate = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };
    const onPreferenceChange = () => {
      if (preference.matches) {
        [orbitRef.current, roomRef.current, roomFillRef.current, progressRef.current, ...copyRefs.current].forEach(node => node?.removeAttribute('style'));
      }
      requestUpdate();
    };
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestUpdate);
    preference.addEventListener('change', onPreferenceChange);
    requestUpdate();
    return () => {
      window.removeEventListener('scroll', requestUpdate);
      window.removeEventListener('resize', requestUpdate);
      preference.removeEventListener('change', onPreferenceChange);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <section ref={sectionRef} id="process" className="portal-story" aria-label="How a photo becomes an explorable space">
      <div className="portal-stage">
        <div className="portal-stage-top" aria-hidden="true">
          <span>01 / THE TRANSFORMATION</span>
          <span>SCROLL TO EXPLORE</span>
        </div>

        <div className="portal-visual" aria-hidden="true">
          <div ref={orbitRef} className="portal-orbit">
            <span className="portal-orbit-outer" />
            <span className="portal-orbit-inner" />
            <span className="portal-orbit-line" />
            <span className="portal-orbit-dot" />
          </div>
          <div ref={roomRef} className="portal-room">
            <svg viewBox="0 0 640 560" role="presentation" focusable="false">
              <defs>
                <linearGradient id="portal-wall" x1="0" x2="1" y1="0" y2="1">
                  <stop stopColor="#a99a86" />
                  <stop offset="1" stopColor="#4b4038" />
                </linearGradient>
                <linearGradient id="portal-floor" x1="0" x2="0" y1="0" y2="1">
                  <stop stopColor="#86715e" />
                  <stop offset="1" stopColor="#2c2521" />
                </linearGradient>
                <linearGradient id="portal-glass" x1="0" x2="1" y1="0" y2="1">
                  <stop stopColor="#e6d5bd" stopOpacity=".54" />
                  <stop offset="1" stopColor="#a98c70" stopOpacity=".16" />
                </linearGradient>
              </defs>
              <g ref={roomFillRef} className="portal-room-fill">
                <path d="M95 79 185 145 185 402 95 478Z" fill="url(#portal-glass)" />
                <path d="M545 79 455 145 455 402 545 478Z" fill="url(#portal-glass)" />
                <path d="M185 145H455V402H185Z" fill="url(#portal-wall)" />
                <path d="M95 478 185 402H455L545 478Z" fill="url(#portal-floor)" />
                <path d="M95 79H545L455 145H185Z" fill="#e6d5bd" fillOpacity=".34" />
                <path d="M271 402V220H369V402" fill="#bfa98c" />
                <path d="M283 402V232H357V402" fill="#f5eadb" />
                <path d="M300 402V282H340V402" fill="#d9bda0" opacity=".5" />
              </g>
              <g fill="none" stroke="#e6d5bd" strokeWidth="1.5" strokeLinejoin="round">
                <path d="M95 79H545V478H95Z" />
                <path d="M185 145H455V402H185Z" />
                <path d="M95 79 185 145M545 79 455 145M95 478 185 402M545 478 455 402" />
                <path d="M271 402V220H369V402" />
              </g>
              <g fill="none" stroke="#c9ae90" strokeWidth="1" opacity=".65">
                <path d="M185 402 126 478M455 402 514 478M230 402 204 478M410 402 436 478M95 478 185 402M545 478 455 402" />
                <path d="M185 402H455M155 427H485M125 452H515" />
              </g>
              <path d="M283 402V232H357V402" fill="none" stroke="#fff" strokeOpacity=".9" strokeWidth="3" />
            </svg>
          </div>
        </div>

        <div className="portal-copy" aria-live="off">
          <div ref={node => { copyRefs.current[0] = node; }} className="portal-copy-panel">
            <p className="portal-eyebrow">A NEW PERSPECTIVE</p>
            <h2>Every space starts <em>somewhere.</em></h2>
            <p>A photograph holds a place in a single frame. I built One Beyond to help you look beyond that frame.</p>
          </div>
          <div ref={node => { copyRefs.current[1] = node; }} className="portal-copy-panel portal-copy-hidden">
            <p className="portal-eyebrow">FROM IMAGE TO GEOMETRY</p>
            <h2>Give the image <em>dimension.</em></h2>
            <p>Walls, floor and objects become a spatial scene you can move through.</p>
          </div>
          <div ref={node => { copyRefs.current[2] = node; }} className="portal-copy-panel portal-copy-hidden">
            <p className="portal-eyebrow">ONE BEYOND</p>
            <h2>Step into <em>the space.</em></h2>
            <p>Upload your photos to begin, or explore a sample reconstruction.</p>
          </div>
        </div>
        <div className="portal-progress" aria-hidden="true"><div ref={progressRef} /></div>
      </div>
    </section>
  );
};
