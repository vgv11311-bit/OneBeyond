import React, { useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowRight } from 'lucide-react';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
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
  const [sceneReady, setSceneReady] = useState(false);

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

  // Reduced motion keeps a still alternative; narrow screens use the same 3D story.
  const staticStory = reduceMotion;

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
      const activeIndex = compact
        ? progress < .08 ? 0 : progress < .27 ? 1 : progress < .49 ? 2 : progress < .74 ? 3 : 4
        : progress < .15 ? 0 : progress < .34 ? 1 : progress < .56 ? 2 : progress < .78 ? 3 : 4;
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
  }, [staticStory, compact]);

  useEffect(() => {
    if (staticStory) return;
    const host = canvasHostRef.current;
    if (!host) return;
    setSceneReady(false);
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
      const camera = new THREE.PerspectiveCamera(compact ? 50 : 42, 1, .1, 80);
      const target = new THREE.Vector3();
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      const ambient = new THREE.HemisphereLight(0xfff5e8, 0x6c5648, 1.7);
      const key = new THREE.DirectionalLight(0xffe6ca, 3.2);
      key.position.set(-4, 8, 5);
      key.castShadow = true;
      key.shadow.mapSize.set(1024, 1024);
      key.shadow.camera.left = -6;
      key.shadow.camera.right = 6;
      key.shadow.camera.top = 6;
      key.shadow.camera.bottom = -6;
      key.shadow.normalBias = .025;
      key.shadow.radius = 3;
      const fill = new THREE.DirectionalLight(0xffffff, .8);
      fill.position.set(5, 5, -4);
      scene.add(ambient, key, fill);

      const warmWall = new THREE.Color('#d5c8b7');
      const blueprintWall = new THREE.Color('#30251d');
      const warmFloor = new THREE.Color('#a9927c');
      const blueprintFloor = new THREE.Color('#181614');
      const floorMaterial = new THREE.MeshStandardMaterial({ color: blueprintFloor, roughness: .92, metalness: 0 });
      const wallMaterial = new THREE.MeshStandardMaterial({ color: blueprintWall, roughness: .88, side: THREE.DoubleSide });
      const woodMaterial = new THREE.MeshStandardMaterial({ color: '#9e7456', roughness: .63 });
      const fabricMaterial = new THREE.MeshStandardMaterial({ color: '#6b5c50', roughness: .98 });
      const cushionMaterial = new THREE.MeshStandardMaterial({ color: '#a58e7a', roughness: 1 });
      const accentMaterial = new THREE.MeshStandardMaterial({ color: '#a95332', roughness: .96 });
      const ivoryMaterial = new THREE.MeshStandardMaterial({ color: '#e6d5bd', roughness: .88 });
      const rugMaterial = new THREE.MeshStandardMaterial({ color: '#c9b8a1', roughness: 1 });
      const leafMaterial = new THREE.MeshStandardMaterial({ color: '#59705e', roughness: .9, side: THREE.DoubleSide });
      const lampShadeMaterial = new THREE.MeshStandardMaterial({ color: '#e7c39b', emissive: '#ffaf63', emissiveIntensity: .5, roughness: .84, side: THREE.DoubleSide });
      const lampBulbMaterial = new THREE.MeshBasicMaterial({ color: '#ffe0ab' });
      const metalMaterial = new THREE.MeshStandardMaterial({ color: '#44382f', metalness: .58, roughness: .42 });
      const windowMaterial = new THREE.MeshStandardMaterial({ color: '#b7c7bf', transparent: true, opacity: .32, metalness: .05, roughness: .25 });
      const lineMaterial = new THREE.LineBasicMaterial({ color: '#e3542a', transparent: true, opacity: 1 });
      const detailLineMaterial = new THREE.LineBasicMaterial({ color: '#d8a88c', transparent: true, opacity: .55 });

      const floor = new THREE.Mesh(new THREE.BoxGeometry(5.4, .06, 6), floorMaterial);
      floor.position.y = -.035;
      floor.receiveShadow = true;
      scene.add(floor);
      const wallSpecs: Array<[number, number, number, number, number]> = [
        [0, -3, 5.4, .08, 3],
        [-2.7, 0, .08, 6, 3],
        [2.7, 0, .08, 6, 3],
      ];
      const walls = wallSpecs.map(([x, z, width, depth, height]) => {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), wallMaterial);
        mesh.position.set(x, 0, z);
        mesh.receiveShadow = true;
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
      const rounded = (x: number, y: number, z: number, width: number, height: number, depth: number, radius: number, material: InstanceType<typeof THREE.MeshStandardMaterial>) => {
        const mesh = new THREE.Mesh(new RoundedBoxGeometry(width, height, depth, 5, radius), material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        furniture.add(mesh);
        return mesh;
      };
      const cylinder = (x: number, y: number, z: number, top: number, bottom: number, height: number, material: InstanceType<typeof THREE.MeshStandardMaterial>) => {
        const mesh = new THREE.Mesh(new THREE.CylinderGeometry(top, bottom, height, 16), material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        furniture.add(mesh);
        return mesh;
      };
      const upholstery = (x: number, y: number, z: number, width: number, height: number, depth: number, material: InstanceType<typeof THREE.MeshStandardMaterial>) => {
        const geometry = new THREE.SphereGeometry(1, 32, 20);
        const points = geometry.attributes.position;
        for (let index = 0; index < points.count; index++) {
          const soften = (value: number) => Math.sign(value) * Math.pow(Math.abs(value), .58);
          points.setXYZ(index, soften(points.getX(index)) * width / 2, soften(points.getY(index)) * height / 2, soften(points.getZ(index)) * depth / 2);
        }
        geometry.computeVertexNormals();
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        furniture.add(mesh);
        return mesh;
      };
      // Upholstery has separate rounded cushions and a continuous curved frame.
      rounded(-.72, .29, -.86, 2.42, .42, 1.02, .16, fabricMaterial);
      const sofaBack = rounded(-.72, .73, -1.30, 2.42, .83, .23, .11, fabricMaterial);
      sofaBack.rotation.x = -.07;
      rounded(-1.85, .52, -.86, .18, .62, 1.02, .085, fabricMaterial);
      rounded(.41, .52, -.86, .18, .62, 1.02, .085, fabricMaterial);
      for (const x of [-1.31, -.13]) {
        const seat = upholstery(x, .55, -.75, 1.06, .22, .78, cushionMaterial);
        seat.rotation.x = -.025;
        const back = upholstery(x, .85, -1.16, 1.02, .48, .22, cushionMaterial);
        back.rotation.x = -.11;
      }
      const throwPillow = upholstery(-1.61, .8, -.91, .36, .39, .16, accentMaterial);
      throwPillow.rotation.z = -.18;
      throwPillow.rotation.y = .22;
      for (const x of [-1.65, .17]) for (const z of [-1.17, -.57]) cylinder(x, .055, z, .035, .025, .11, metalMaterial);

      // A woven rug and a low, oval coffee table give the room softer edges.
      rounded(1.05, .014, .55, 3.1, .025, 2.25, .012, rugMaterial).castShadow = false;
      const tableTop = new THREE.Mesh(new THREE.CylinderGeometry(.61, .61, .09, 48), woodMaterial);
      tableTop.scale.z = .62;
      tableTop.position.set(1.1, .43, .5);
      tableTop.castShadow = true;
      tableTop.receiveShadow = true;
      furniture.add(tableTop);
      for (const x of [.74, 1.46]) for (const z of [.29, .71]) cylinder(x, .2, z, .022, .03, .4, metalMaterial);
      // The table lamp rests on the top and gives the seating area a warm pool of light.
      cylinder(1.1, .505, .5, .075, .085, .055, ivoryMaterial);
      cylinder(1.1, .605, .5, .013, .013, .16, metalMaterial);
      const lampShade = new THREE.Mesh(new THREE.CylinderGeometry(.085, .145, .17, 24, 1, true), lampShadeMaterial);
      lampShade.position.set(1.1, .75, .5);
      lampShade.castShadow = true;
      furniture.add(lampShade);
      const lampBulb = new THREE.Mesh(new THREE.SphereGeometry(.045, 12, 8), lampBulbMaterial);
      lampBulb.position.set(1.1, .69, .5);
      furniture.add(lampBulb);
      const lampGlow = new THREE.PointLight(0xffb86d, 1.5, 2.1, 2);
      lampGlow.position.set(1.1, .69, .5);
      furniture.add(lampGlow);

      // The top joins the cabinet. Every display object rests on a shelf.
      rounded(1.85, .92, -2.72, .86, 1.72, .32, .045, woodMaterial);
      for (const y of [.48, 1.01, 1.54]) rounded(1.85, y, -2.48, .78, .055, .45, .025, ivoryMaterial);
      rounded(1.85, 1.79, -2.52, .82, .06, .4, .025, ivoryMaterial);
      for (const [x, height, material] of [
        [1.63, .26, accentMaterial], [1.76, .32, woodMaterial], [1.89, .22, cushionMaterial],
      ] as const) rounded(x, .51 + height / 2, -2.39, .1, height, .18, .018, material);
      cylinder(1.68, 1.15, -2.37, .075, .11, .23, ivoryMaterial);
      rounded(2.02, 1.17, -2.39, .19, .26, .06, .015, metalMaterial);
      const displayOrb = new THREE.Mesh(new THREE.SphereGeometry(.105, 20, 12), accentMaterial);
      displayOrb.position.set(1.69, 1.67, -2.39);
      displayOrb.castShadow = true;
      furniture.add(displayOrb);
      rounded(2.01, 1.625, -2.39, .17, .12, .17, .035, woodMaterial);
      cylinder(1.85, 1.92, -2.52, .11, .08, .2, accentMaterial);
      cylinder(1.85, 2.105, -2.52, .012, .012, .18, leafMaterial);
      for (const direction of [-1, 1]) {
        const leaf = new THREE.Mesh(new THREE.SphereGeometry(.13, 12, 8), leafMaterial);
        leaf.scale.set(.95, .18, .38);
        leaf.position.set(1.85 + direction * .11, 2.11, -2.52);
        leaf.rotation.z = direction * .35;
        leaf.castShadow = true;
        furniture.add(leaf);
      }
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
        const rise = smooth(span(progress, compact ? .04 : .15, compact ? .42 : .52));
        const finish = smooth(span(progress, compact ? .32 : .48, compact ? .76 : .80));
        const walkthrough = smooth(span(progress, .72, 1));
        outlineGeometry.setDrawRange(0, Math.max(2, Math.ceil(span(progress, compact ? .01 : .15, compact ? .18 : .30) * outlinePoints.length)));
        lineMaterial.opacity = 1 - finish * .68;
        detailLineMaterial.opacity = .12 + (1 - finish) * .55;
        floorMaterial.color.copy(blueprintFloor).lerp(warmFloor, finish);
        wallMaterial.color.copy(blueprintWall).lerp(warmWall, finish);
        walls.forEach((wall) => {
          wall.scale.y = Math.max(.001, rise);
          wall.position.y = 1.5 * rise;
        });
        furniture.scale.y = Math.max(.001, smooth(span(progress, compact ? .27 : .40, compact ? .65 : .70)));
        furniture.visible = progress > (compact ? .27 : .40);
        planFurniture.visible = progress < (compact ? .63 : .68);
        planTable.visible = progress < (compact ? .63 : .68);
        camera.position.set(
          (compact ? 6.4 : 5.9) * (1 - walkthrough),
          (compact ? 7.5 : 8.6) - (compact ? 5.86 : 6.96) * walkthrough,
          (compact ? 9.4 : 8.0) - (compact ? 6.78 : 5.38) * walkthrough,
        );
        target.set(0, .12 + 1.27 * walkthrough, -.8 * walkthrough);
        camera.lookAt(target);
        renderer.render(scene, camera);
      };
      sceneUpdateRef.current = updateScene;
      const observer = new ResizeObserver(resize);
      observer.observe(host);
      resize();
      setSceneReady(true);
      cleanup = () => {
        observer.disconnect();
        sceneUpdateRef.current = null;
        scene.traverse((object) => {
          if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.LineSegments) {
            object.geometry.dispose();
          }
        });
        [floorMaterial, wallMaterial, woodMaterial, fabricMaterial, cushionMaterial, accentMaterial, ivoryMaterial, rugMaterial, leafMaterial, lampShadeMaterial, lampBulbMaterial, metalMaterial, windowMaterial, lineMaterial, detailLineMaterial].forEach((material) => material.dispose());
        renderer.dispose();
        renderer.domElement.remove();
      };
    }).catch(() => { if (!disposed) setSceneFailed(true); });
    return () => { disposed = true; cleanup(); };
  }, [staticStory, compact]);

  return <section ref={sectionRef} id="home" className={`blueprint-story${staticStory ? ' blueprint-reduced' : ''}`} aria-label="From image to explorable space">
    <div className="blueprint-stage">
      {!staticStory && !sceneFailed && <div ref={canvasHostRef} className="blueprint-canvas" aria-hidden="true" />}
      {(staticStory || sceneFailed) && !compact && <div className="blueprint-fallback" aria-hidden="true"><svg viewBox="0 0 540 480" focusable="false"><path d="M70 65H470V415H70Z M140 130H400V350H140Z M70 65 140 130M470 65 400 130M70 415 140 350M470 415 400 350" /><path d="M140 350H400M210 350V220H330V350M210 220H330" /></svg></div>}
      {compact && (staticStory || sceneFailed || !sceneReady) && <div className="blueprint-fallback blueprint-mobile-fallback" aria-hidden="true"><svg viewBox="0 0 540 480" focusable="false">
        <path className="blueprint-plan-floor" d="M55 225 270 80 485 205 270 355Z" />
        <path d="M55 225 270 80 485 205M55 225v-85L270 8l215 125v72M270 80v-72M270 355 485 205" />
        <path className="blueprint-plan-furniture" d="m134 215 112-72 85 48-112 75Zm22-10 89-57m-55 104 113-73m40 34 58-39 43 25-58 39Zm-24 29 44 25-44 28-44-25Z" />
        <path className="blueprint-plan-window" d="m325 50 76 44v48l-76-44Z" />
      </svg></div>}
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
