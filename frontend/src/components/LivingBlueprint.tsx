import React, { useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowRight } from 'lucide-react';
import { navigateSite } from '../siteRoutes';

interface LivingBlueprintProps {
  onOpenUpload: () => void;
}

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const span = (value: number, start: number, end: number) => clamp((value - start) / (end - start));
const smooth = (value: number) => value * value * (3 - 2 * value);

/** An authored illustration of spatial reconstruction, never a user result. */
export const LivingBlueprint: React.FC<LivingBlueprintProps> = ({ onOpenUpload }) => {
  const sectionRef = useRef<HTMLElement>(null);
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const copyRefs = useRef<(HTMLDivElement | null)[]>([]);
  const progressBarRef = useRef<HTMLDivElement>(null);
  const sceneUpdateRef = useRef<((progress: number) => void) | null>(null);
  const currentProgressRef = useRef(0);
  const [reduceMotion, setReduceMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [compact, setCompact] = useState(() => window.matchMedia('(max-width: 760px)').matches);
  const [sceneFailed, setSceneFailed] = useState(false);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduceMotion(preference.matches);
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    const viewport = window.matchMedia('(max-width: 760px)');
    const update = () => setCompact(viewport.matches);
    viewport.addEventListener('change', update);
    return () => viewport.removeEventListener('change', update);
  }, []);

  const staticStory = reduceMotion || compact;

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const section = sectionRef.current;
      if (!section) return;
      const travel = Math.max(1, section.offsetHeight - window.innerHeight);
      const progress = clamp(-section.getBoundingClientRect().top / travel);
      currentProgressRef.current = progress;
      sceneUpdateRef.current?.(progress);
      if (progressBarRef.current) progressBarRef.current.style.transform = `scaleX(${progress})`;
      if (staticStory) return;
      const activeIndex = progress < .15 ? 0 : progress < .34 ? 1 : progress < .56 ? 2 : progress < .78 ? 3 : 4;
      copyRefs.current.forEach((node, index) => {
        if (!node) return;
        const active = index === activeIndex;
        node.style.opacity = active ? '1' : '0';
        node.style.transform = 'none';
        node.style.pointerEvents = active ? 'auto' : 'none';
        node.inert = !active;
      });
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(update); };
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    schedule();
    return () => {
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [staticStory]);

  useEffect(() => {
    if (staticStory) return;
    const host = canvasHostRef.current;
    if (!host) return;
    let disposed = false;
    let cleanup = () => {};

    void import('three').then((THREE) => {
      if (disposed) return;
      let renderer: InstanceType<typeof THREE.WebGLRenderer>;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
      } catch {
        setSceneFailed(true);
        return;
      }
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.3;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, window.innerWidth < 720 ? 1 : 1.5));
      host.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(42, 1, .1, 80);
      const target = new THREE.Vector3();
      const ambient = new THREE.AmbientLight(0xffffff, 2.1);
      const key = new THREE.DirectionalLight(0xffe6ca, 3.0);
      key.position.set(-4, 8, 5);
      const fill = new THREE.DirectionalLight(0xffffff, 1.3);
      fill.position.set(5, 5, -4);
      scene.add(ambient, key, fill);

      const warmWall = new THREE.Color('#d5c8b7');
      const blueprintWall = new THREE.Color('#30251d');
      const warmFloor = new THREE.Color('#a9927c');
      const blueprintFloor = new THREE.Color('#181614');
      const floorMaterial = new THREE.MeshStandardMaterial({ color: blueprintFloor, roughness: .92, metalness: 0 });
      const wallMaterial = new THREE.MeshStandardMaterial({ color: blueprintWall, roughness: .88, side: THREE.DoubleSide });
      const woodMaterial = new THREE.MeshStandardMaterial({ color: '#997359', roughness: .75 });
      const fabricMaterial = new THREE.MeshStandardMaterial({ color: '#5c5047', roughness: .95 });
      const ivoryMaterial = new THREE.MeshStandardMaterial({ color: '#e6d5bd', roughness: .88 });
      const metalMaterial = new THREE.MeshStandardMaterial({ color: '#44382f', metalness: .58, roughness: .42 });
      const windowMaterial = new THREE.MeshStandardMaterial({ color: '#b7c7bf', transparent: true, opacity: .32, metalness: .05, roughness: .25 });
      const lineMaterial = new THREE.LineBasicMaterial({ color: '#e3542a', transparent: true, opacity: 1 });
      const detailLineMaterial = new THREE.LineBasicMaterial({ color: '#d8a88c', transparent: true, opacity: .55 });

      const floor = new THREE.Mesh(new THREE.BoxGeometry(5.4, .06, 6), floorMaterial);
      floor.position.y = -.035;
      scene.add(floor);
      const wallSpecs: Array<[number, number, number, number, number]> = [
        [0, -3, 5.4, .08, 3],
        [-2.7, 0, .08, 6, 3],
        [2.7, 0, .08, 6, 3],
      ];
      const walls = wallSpecs.map(([x, z, width, depth, height]) => {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), wallMaterial);
        mesh.position.set(x, 0, z);
        scene.add(mesh);
        return mesh;
      });

      const outlinePoints = [
        [-2.7, .035, 3], [-2.7, .035, -3], [2.7, .035, -3], [2.7, .035, 3], [-2.7, .035, 3],
      ];
      const outlineGeometry = new THREE.BufferGeometry().setFromPoints(outlinePoints.map(([x, y, z]) => new THREE.Vector3(x, y, z)));
      const outline = new THREE.Line(outlineGeometry, lineMaterial);
      scene.add(outline);
      const gridPoints: InstanceType<typeof THREE.Vector3>[] = [];
      for (let i = -2; i <= 2; i++) {
        gridPoints.push(new THREE.Vector3(i, .03, -3), new THREE.Vector3(i, .03, 3));
        gridPoints.push(new THREE.Vector3(-2.7, .03, i), new THREE.Vector3(2.7, .03, i));
      }
      const grid = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gridPoints), detailLineMaterial);
      scene.add(grid);

      const furniture = new THREE.Group();
      scene.add(furniture);
      const box = (x: number, y: number, z: number, width: number, height: number, depth: number, material: InstanceType<typeof THREE.MeshStandardMaterial>) => {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
        mesh.position.set(x, y, z);
        furniture.add(mesh);
        return mesh;
      };
      box(-.72, .28, -.86, 2.35, .42, .92, fabricMaterial);
      box(-.72, .72, -1.23, 2.35, .72, .20, fabricMaterial);
      box(-1.85, .52, -.86, .18, .7, .92, fabricMaterial);
      box(.40, .52, -.86, .18, .7, .92, fabricMaterial);
      box(-1.31, .52, -.80, 1.05, .13, .68, ivoryMaterial);
      box(-.13, .52, -.80, 1.05, .13, .68, ivoryMaterial);
      box(1.2, .33, .45, 1.15, .12, .65, woodMaterial);
      box(.77, .17, .18, .06, .28, .06, metalMaterial);
      box(1.63, .17, .18, .06, .28, .06, metalMaterial);
      box(.77, .17, .72, .06, .28, .06, metalMaterial);
      box(1.63, .17, .72, .06, .28, .06, metalMaterial);
      box(1.85, 1.04, -2.73, .82, 1.62, .24, woodMaterial);
      box(1.85, .78, -2.55, .7, .06, .48, ivoryMaterial);
      box(1.85, 1.34, -2.55, .7, .06, .48, ivoryMaterial);
      const windowPane = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 1.42), windowMaterial);
      windowPane.position.set(-.62, 1.9, -2.9);
      furniture.add(windowPane);
      const windowFrame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1.62, 1.44, .04)), detailLineMaterial);
      windowFrame.position.set(-.62, 1.9, -2.87);
      furniture.add(windowFrame);
      const planFurniture = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.35, .01, .92)), detailLineMaterial);
      planFurniture.position.set(-.72, .045, -.86);
      scene.add(planFurniture);
      const planTable = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1.15, .01, .65)), detailLineMaterial);
      planTable.position.set(1.2, .045, .45);
      scene.add(planTable);

      const resize = () => {
        const width = host.clientWidth || window.innerWidth;
        const height = host.clientHeight || window.innerHeight;
        camera.aspect = width / Math.max(1, height);
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
        sceneUpdateRef.current?.(currentProgressRef.current);
      };
      const updateScene = (progress: number) => {
        const rise = smooth(span(progress, .15, .52));
        const finish = smooth(span(progress, .48, .80));
        const walkthrough = smooth(span(progress, .72, 1));
        outlineGeometry.setDrawRange(0, Math.max(2, Math.ceil(span(progress, .15, .30) * outlinePoints.length)));
        lineMaterial.opacity = 1 - finish * .68;
        detailLineMaterial.opacity = .12 + (1 - finish) * .55;
        floorMaterial.color.copy(blueprintFloor).lerp(warmFloor, finish);
        wallMaterial.color.copy(blueprintWall).lerp(warmWall, finish);
        walls.forEach((wall) => {
          wall.scale.y = Math.max(.001, rise);
          wall.position.y = 1.5 * rise;
        });
        furniture.scale.y = Math.max(.001, smooth(span(progress, .40, .70)));
        furniture.visible = progress > .40;
        planFurniture.visible = progress < .68;
        planTable.visible = progress < .68;
        camera.position.set(5.9 * (1 - walkthrough), 8.6 - 6.96 * walkthrough, 8.0 - 5.38 * walkthrough);
        target.set(0, .12 + 1.27 * walkthrough, -.8 * walkthrough);
        camera.lookAt(target);
        renderer.render(scene, camera);
      };
      sceneUpdateRef.current = updateScene;
      const observer = new ResizeObserver(resize);
      observer.observe(host);
      resize();
      cleanup = () => {
        observer.disconnect();
        sceneUpdateRef.current = null;
        scene.traverse((object) => {
          if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.LineSegments) {
            object.geometry.dispose();
          }
        });
        [floorMaterial, wallMaterial, woodMaterial, fabricMaterial, ivoryMaterial, metalMaterial, windowMaterial, lineMaterial, detailLineMaterial].forEach((material) => material.dispose());
        renderer.dispose();
        renderer.domElement.remove();
      };
    }).catch(() => { if (!disposed) setSceneFailed(true); });
    return () => { disposed = true; cleanup(); };
  }, [staticStory]);

  return <section ref={sectionRef} id="home" className={`blueprint-story${staticStory ? ' blueprint-reduced' : ''}`} aria-label="From image to explorable space">
    <div className="blueprint-stage">
      {!staticStory && !sceneFailed && <div ref={canvasHostRef} className="blueprint-canvas" aria-hidden="true" />}
      {(staticStory || sceneFailed) && <div className="blueprint-fallback" aria-hidden="true"><svg viewBox="0 0 540 480" focusable="false"><path d="M70 65H470V415H70Z M140 130H400V350H140Z M70 65 140 130M470 65 400 130M70 415 140 350M470 415 400 350" /><path d="M140 350H400M210 350V220H330V350M210 220H330" /></svg></div>}
      <div className="blueprint-grain" aria-hidden="true" />
      <div className="blueprint-top" aria-hidden="true"><span>ONE BEYOND / LIVING BLUEPRINT</span><span>ILLUSTRATIVE SPATIAL STORY</span></div>
      <div className="blueprint-copies">
        <div ref={node => { copyRefs.current[0] = node; }} className="blueprint-copy blueprint-copy-intro">
          <p className="section-kicker">REAL PLACES, NEW PERSPECTIVES</p>
          <h1 className="font-display">See a place <em>beyond the frame.</em></h1>
          <p>From photographs of a real space to a room you can explore.</p>
          <div className="blueprint-actions"><button type="button" onClick={onOpenUpload} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Create 3D Space <ArrowRight size={16} aria-hidden="true" /></button><button type="button" onClick={() => navigateSite('sample')} className="aethera-text-button rounded-full px-4 py-3 text-sm font-medium">Explore a sample</button></div>
        </div>
        <div ref={node => { copyRefs.current[1] = node; }} className="blueprint-copy blueprint-copy-hidden" inert><p className="section-kicker">01 / TRACE</p><h2 className="font-display">Find the <em>shape.</em></h2><p>A line marks the floor and establishes the space.</p></div>
        <div ref={node => { copyRefs.current[2] = node; }} className="blueprint-copy blueprint-copy-hidden" inert><p className="section-kicker">02 / BUILD</p><h2 className="font-display">Raise the <em>room.</em></h2><p>Boundaries rise; depth replaces the flat plan.</p></div>
        <div ref={node => { copyRefs.current[3] = node; }} className="blueprint-copy blueprint-copy-hidden" inert><p className="section-kicker">03 / REVEAL</p><h2 className="font-display">Bring it <em>to life.</em></h2><p>Objects, surfaces and light give the space character.</p></div>
        <div ref={node => { copyRefs.current[4] = node; }} className="blueprint-copy blueprint-copy-hidden" inert><p className="section-kicker">04 / EXPLORE</p><h2 className="font-display">Step <em>inside.</em></h2><p>This animation is illustrative. Open the sample to explore an actual room in the viewer.</p><button type="button" onClick={() => navigateSite('sample')} className="aethera-button contact-button rounded-full px-7 py-3.5 text-sm font-medium">Enter Sample Viewer <ArrowRight size={16} aria-hidden="true" /></button></div>
      </div>
      {!staticStory && <div className="blueprint-scroll-cue" aria-hidden="true">SCROLL TO CONTINUE <ArrowDown size={14} /></div>}
      <div className="blueprint-progress" aria-hidden="true"><div ref={progressBarRef} /></div>
    </div>
  </section>;
};
