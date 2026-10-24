import React, { useState, Suspense, useRef, useEffect, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { AdaptiveDpr } from '@react-three/drei';
import { WebGPURenderer } from 'three/webgpu';
import { RoomModel } from './RoomModel';
import { ViewerErrorBoundary } from './ViewerErrorBoundary';
import { furnitureObstacles, moveWithinRoom, Obstacle, Vec3 } from './walkthrough';
import { WindowDaylight } from './WindowDaylight';
import { FirstPersonControls } from './FirstPersonControls';
import { SpatialHUD } from './SpatialHUD';
import { ViewerToolbar } from './ViewerToolbar';
import { RenderDiagnostics } from './RenderDiagnostics';
import { HighFidelityPipeline } from './HighFidelityPipeline';
import {
  GRAPHICS_PROFILES,
  detectGraphicsCapabilities,
  resolveGraphicsQuality,
  saveRenderBaseline,
} from './graphicsProfiles';
import {
  SpaceJob,
  LightingMode,
  InferredLighting,
  GraphicsCapabilities,
  GraphicsQuality,
  RenderMetrics,
} from '../../types';
import { Box } from 'lucide-react';
import * as THREE from 'three';
import { photoVerticalFov } from './photoFraming';

interface SpatialViewerProps {
  job: SpaceJob;
  onExit: () => void;
}

// Match the reconstruction's lens without touching the walking pose. A fixed
// 68-degree lens made fitted objects look displaced in the delivered viewer.
const PhotoLens: React.FC<{ calibration?: InferredLighting['projection_camera'] }> = ({ calibration }) => {
  const { camera, size } = useThree();
  useEffect(() => {
    if (!(camera instanceof THREE.PerspectiveCamera) || !calibration) return;
    const fy = calibration.intrinsics?.[1]?.[1];
    const [h, w] = calibration.source_shape;
    if (!(fy > 0 && h > 0 && w > 0)) return;
    // Preserve the original contain framing without cropping on wide screens.
    const photoAspect = w / h;
    const viewportAspect = size.width / Math.max(1, size.height);
    camera.fov = photoVerticalFov(fy, photoAspect, viewportAspect);
    camera.updateProjectionMatrix();
  }, [camera, size.width, size.height, calibration]);
  return null;
};

const PhotoFixture: React.FC<{position: [number,number,number]; intensity:number; color:THREE.Color; distance:number; decay:number; ceiling:boolean}> = ({ceiling,...props}) => {
  const target=React.useMemo(() => {
    const object=new THREE.Object3D();
    object.position.set(props.position[0],0,props.position[2]);
    return object;
  },[props.position[0],props.position[2]]);
  // Recessed downlights must illuminate downward, not burn the ceiling above.
  return ceiling ? <><primitive object={target}/><spotLight {...props} target={target} angle={1.05} penumbra={.8}/></> : <pointLight {...props}/>;
};

const LightingRig: React.FC<{
  mode: LightingMode;
  realTimeShadows: boolean;
  isSample: boolean;
  inferred?: InferredLighting;
  objects: Array<Record<string, any>>;
  roomBounds: [number, number, number];
  shadowMapSize: 256 | 512 | 1024;
  maxDynamicLights: number;
}> = ({ mode, realTimeShadows, isSample, inferred, objects, roomBounds, shadowMapSize, maxDynamicLights }) => {
  if (mode === 'cinematic') {
    if (!isSample && inferred) {
      // Photo-derived materials already contain the source's white balance.
      // A second whole-room warm/cool tint shifts every sampled material.
      // Keep broad illumination neutral; measured fixtures retain local color.
      const neutral = new THREE.Color('#ffffff');
      const keyColor = neutral;
      const daylight = inferred.daylight;
      const [imageX, imageY] = inferred.key_image_position || [0.5, 0.2];
      const keyPosition: [number, number, number] = [
        (imageX - 0.5) * roomBounds[0] * 1.8,
        Math.max(roomBounds[2] * 0.75, (1.15 - imageY) * roomBounds[2]),
        Math.min(1.2, roomBounds[1] * 0.12),
      ];
      const lamps = objects
        .filter((object) => /lamp|light|chandelier/.test(String(object.label || '').toLowerCase()))
        .sort((a,b) => Number(a.label==='ceiling_lamp')-Number(b.label==='ceiling_lamp') || (b.fixture_light?.intensity ?? 0)-(a.fixture_light?.intensity ?? 0))
        .slice(0, maxDynamicLights);
      return (
        <>
          <ambientLight intensity={daylight ? .18 + inferred.ambient_intensity * .22 : .35 + inferred.ambient_intensity * .55} color={neutral} />
          <hemisphereLight intensity={daylight ? .12 : .15 + inferred.brightness * .20} color={neutral} groundColor="#403834" />
          {daylight?.windows.map((window,index) => <WindowDaylight key={index} window={window}
            room={roomBounds} shadows={realTimeShadows} mapSize={shadowMapSize} />)}
          {!daylight && <directionalLight
            position={keyPosition}
            intensity={inferred.key_intensity}
            color={keyColor}
            castShadow={realTimeShadows}
            shadow-mapSize-width={shadowMapSize}
            shadow-mapSize-height={shadowMapSize}
            shadow-camera-near={0.2}
            shadow-camera-far={Math.max(16, roomBounds[1] * 2)}
            shadow-camera-left={-roomBounds[0]}
            shadow-camera-right={roomBounds[0]}
            shadow-camera-top={roomBounds[2] * 2}
            shadow-camera-bottom={-roomBounds[2]}
            shadow-bias={-0.0003}
            shadow-normalBias={0.008}
          />}
          {!daylight && <directionalLight
            position={[-keyPosition[0] * 0.7, roomBounds[2] * 0.8, -roomBounds[1] * 0.7]}
            intensity={0.16 + inferred.brightness * 0.25}
            color={neutral}
          />}
          {lamps.map((lamp, index) => (
            <PhotoFixture
              key={`${lamp.label}-${index}`}
              position={(lamp.fixture_light?.position || lamp.center || [0, roomBounds[2] * 0.75, -roomBounds[1] / 2]) as [number, number, number]}
              intensity={lamp.fixture_light?.intensity ?? (0.28 + inferred.brightness * 0.55)}
              color={lamp.fixture_light?.color ? new THREE.Color().setRGB(...lamp.fixture_light.color as [number,number,number], THREE.SRGBColorSpace) : keyColor}
              distance={Math.max(4, roomBounds[0] * 1.2)}
              decay={2}
              ceiling={lamp.label==='ceiling_lamp' || lamp.label==='chandelier'}
            />
          ))}
        </>
      );
    }
    return (
      <>
        {/* Broad sky fill keeps the warm sunlit room readable rather than
            turning the far wall into a black void. */}
        <ambientLight intensity={0.42} color="#f4eadc" />
        {/* Warm directional daylight enters from the sample room's left window. */}
        <directionalLight
          position={[-4, 5, -4]}
          intensity={1.75}
          color="#fff0c8"
          castShadow={realTimeShadows}
          shadow-mapSize-width={shadowMapSize}
          shadow-mapSize-height={shadowMapSize}
          shadow-camera-near={0.5}
          shadow-camera-far={18}
          shadow-camera-left={-7}
          shadow-camera-right={7}
          shadow-camera-top={7}
          shadow-camera-bottom={-7}
          shadow-bias={-0.0002}
          shadow-normalBias={0.025}
        />
        {/* Soft bounced daylight from the opposite window wall keeps the
            complete room readable while the warm key still suggests sun. */}
        <directionalLight position={[4, 6, 3]} intensity={0.28} color="#ffe8c2" />
        <directionalLight position={[-4, 4, 3]} intensity={0.18} color="#cde5ff" />
        <pointLight position={[0.15, 2.15, -4.35]} intensity={1.1} color="#ffe6a8" distance={5} />
        <pointLight position={[1.25, 1.63, -5.9]} intensity={0.52} color="#ffe5ad" distance={3} />
        <pointLight position={[-2.55, 1.25, -3.15]} intensity={0.38} color="#ffe2ac" distance={2.2} />
      </>
    );
  }

  if (mode === 'studio') {
    return (
      <>
        <ambientLight intensity={0.85} color="#ffffff" />
        <directionalLight position={[4, 6, 3]} intensity={1.0} color="#ffffff" castShadow={realTimeShadows} shadow-mapSize-width={shadowMapSize} shadow-mapSize-height={shadowMapSize} shadow-bias={-0.0002} shadow-normalBias={0.025} />
        <directionalLight position={[-4, 4, -3]} intensity={0.6} color="#e0f2fe" />
      </>
    );
  }

  if (mode === 'daylight') {
    return (
      <>
        <ambientLight intensity={0.7} color="#fff8e7" />
        <directionalLight position={[5, 4, 0]} intensity={2.2} color="#fff1cc" castShadow={realTimeShadows} shadow-mapSize-width={shadowMapSize} shadow-mapSize-height={shadowMapSize} shadow-bias={-0.0002} shadow-normalBias={0.025} />
        <hemisphereLight groundColor="#222" color="#87ceeb" intensity={0.6} />
      </>
    );
  }

  // wireframe
  return (
    <>
      <ambientLight intensity={1.5} color="#00f0ff" />
    </>
  );
};

// Virtual joystick mobile controller within Three.js context
const MobileJoystickDriver: React.FC<{
  virtualMove: { dx: number; dy: number };
  obstacles: Obstacle[];
  virtualHeight: number;
  roomBounds: [number, number, number];
  roomLimits: [number, number, number, number];
}> = ({ virtualMove, virtualHeight, roomBounds, roomLimits, obstacles }) => {
  const { camera, performance: renderPerformance } = useThree();
  const forwardRef = useRef(new THREE.Vector3());
  const rightRef = useRef(new THREE.Vector3());
  const moveRef = useRef(new THREE.Vector3());
  const [hw, hl, hh] = [roomBounds[0] / 2 - 0.3, roomBounds[1] / 2 - 0.3, roomBounds[2] - 0.2];

  useFrame((_, delta) => {
    delta = Math.min(delta, 0.05);
    if (Math.abs(virtualMove.dx) > 0.05 || Math.abs(virtualMove.dy) > 0.05) {
      renderPerformance.regress();
      const forward = forwardRef.current.set(0, 0, -1).applyQuaternion(camera.quaternion);
      forward.y = 0;
      forward.normalize();

      const right = rightRef.current.set(1, 0, 0).applyQuaternion(camera.quaternion);
      right.y = 0;
      right.normalize();

      const move = moveRef.current.set(0, 0, 0);
      move.addScaledVector(forward, virtualMove.dy);
      move.addScaledVector(right, virtualMove.dx);

      move.clampLength(0, 1);
      camera.position.fromArray(moveWithinRoom(camera.position.toArray() as Vec3,
        [move.x * delta * 3.5,0,move.z * delta * 3.5],roomLimits,roomBounds[2],obstacles));
      camera.position.x = Math.max(roomLimits[0], Math.min(roomLimits[1], camera.position.x));
      camera.position.z = Math.max(roomLimits[2], Math.min(roomLimits[3], camera.position.z));
    }

    if (virtualHeight !== 0) {
      renderPerformance.regress();
      camera.position.fromArray(moveWithinRoom(camera.position.toArray() as Vec3,
        [0,virtualHeight * delta * 2.0,0],roomLimits,roomBounds[2],obstacles));
      camera.position.y = Math.max(0.4, Math.min(hh, camera.position.y));
    }
  });

  return null;
};

export const SpatialViewer: React.FC<SpatialViewerProps> = ({ job, onExit }) => {
  const [clearView, setClearView] = useState(false);
  const [lightingMode, setLightingMode] = useState<LightingMode>('cinematic');
  const [graphicsQuality, setGraphicsQuality] = useState<GraphicsQuality>(() => {
    const saved = localStorage.getItem('one-beyond:graphics-quality');
    return saved === 'high' || saved === 'balanced' || saved === 'compatibility' ? saved : 'auto';
  });
  const [graphicsCapabilities, setGraphicsCapabilities] = useState<GraphicsCapabilities>();
  const [renderMetrics, setRenderMetrics] = useState<RenderMetrics>();
  const [baselineCaptured, setBaselineCaptured] = useState(false);
  const [showPointCloud, setShowPointCloud] = useState(false);
  const [isPointerLocked, setIsPointerLocked] = useState(false);
  const [virtualMove, setVirtualMove] = useState({ dx: 0, dy: 0 });
  const [virtualHeightDelta, setVirtualHeightDelta] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [cameraKey, setCameraKey] = useState(0);

  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const viewerStartedAtRef = useRef(performance.now());
  const [modelLoadMs, setModelLoadMs] = useState<number>();

  useEffect(() => {
    let active = true;
    detectGraphicsCapabilities().then((capabilities) => {
      if (active) setGraphicsCapabilities(capabilities);
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (graphicsQuality === 'auto') {
      localStorage.removeItem('one-beyond:graphics-quality');
    } else {
      localStorage.setItem('one-beyond:graphics-quality', graphicsQuality);
    }
    setBaselineCaptured(false);
  }, [graphicsQuality]);

  useEffect(() => {
    const updatePointerLock = () => {
      const canvas = canvasContainerRef.current?.querySelector('canvas');
      setIsPointerLocked(document.pointerLockElement === canvas);
    };
    document.addEventListener('pointerlockchange', updatePointerLock);
    document.addEventListener('pointerlockerror', updatePointerLock);
    return () => {
      document.removeEventListener('pointerlockchange', updatePointerLock);
      document.removeEventListener('pointerlockerror', updatePointerLock);
    };
  }, []);

  const roomWidth = job.metrics?.estimated_room_width || 6.0;
  const roomLength = job.metrics?.estimated_room_length || 7.0;
  const roomHeight = job.metrics?.estimated_room_height || 3.0;
  const isSample = job.input_type === 'sample';
  const fallbackCapabilities: GraphicsCapabilities = {
    webgpuAvailable: false,
    webgl2Available: true,
    webglAvailable: true,
    maxTextureSize: 8192,
    logicalCores: 4,
  };
  const resolvedGraphicsQuality = resolveGraphicsQuality(
    graphicsQuality,
    graphicsCapabilities || fallbackCapabilities,
  );
  const graphicsProfile = GRAPHICS_PROFILES[resolvedGraphicsQuality];
  const roomBounds: [number, number, number] = [roomWidth, roomLength, roomHeight];
  const inferredLighting = job.metrics?.extra_details?.lighting as InferredLighting | undefined;
  const sceneObjects = (job.metrics?.extra_details?.scene_objects || []) as Array<Record<string, any>>;
  // The generated room starts at the camera entrance (z≈0) and extends away
  // from it. Use those real limits instead of a symmetric half-length cap.
  const roomLimits: [number, number, number, number] = [
    -roomWidth / 2 + 0.22,
    roomWidth / 2 - 0.22,
    -roomLength + 0.32,
    0.02,
  ];

  const glbUrl = job.glb_url || `/api/spaces/${job.job_id}/model.glb`;
  const toneMappingExposure = !isSample && inferredLighting && Number.isFinite(inferredLighting.exposure)
    // Respect photo-derived exposure instead of darkening every upload by
    // another 14% and clipping bright photographs to a low fixed ceiling.
    ? Math.max(0.72, Math.min(1.4, inferredLighting.exposure * 1.05))
    : 0.95;
  const useUniversalRenderer = Boolean(
    graphicsCapabilities?.webgpuAvailable || graphicsCapabilities?.webgl2Available,
  );

  const createRenderer = useCallback(async (props: { canvas: HTMLCanvasElement }) => {
    if (graphicsCapabilities?.webgpuAvailable || graphicsCapabilities?.webgl2Available) {
      const renderer = new WebGPURenderer({
        canvas: props.canvas,
        // Screen-space passes read a single-sample depth buffer. FXAA is
        // applied at the end of the node pipeline for high/balanced modes.
        antialias: false,
        alpha: false,
        powerPreference: 'high-performance',
        forceWebGL: !graphicsCapabilities.webgpuAvailable,
      });
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = toneMappingExposure;
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      await renderer.init();
      return renderer;
    }

    const renderer = new THREE.WebGLRenderer({
      canvas: props.canvas,
      antialias: graphicsProfile.antialias,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = toneMappingExposure;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    return renderer;
  }, [graphicsCapabilities, graphicsProfile.antialias, toneMappingExposure]);

  const handleLockPointer = () => {
    const canvas = canvasContainerRef.current?.querySelector('canvas');
    if (canvas?.isConnected && canvas.ownerDocument === document && canvas.requestPointerLock) {
      try {
        const request = canvas.requestPointerLock();
        if (request && typeof request.catch === 'function') {
          request.catch(() => setIsPointerLocked(false));
        }
      } catch {
        setIsPointerLocked(false);
      }
    }
  };

  const handleResetCamera = () => {
    setCameraKey((prev) => prev + 1);
  };

  const handleModelLoaded = () => {
    if (!modelLoaded) {
      setModelLoadMs(Math.round(performance.now() - viewerStartedAtRef.current));
    }
    setModelLoaded(true);
  };

  const handleCaptureBaseline = () => {
    if (!graphicsCapabilities || !renderMetrics) return;
    saveRenderBaseline({
      version: 1,
      capturedAt: new Date().toISOString(),
      jobId: job.job_id,
      requestedQuality: graphicsQuality,
      resolvedQuality: resolvedGraphicsQuality,
      capabilities: graphicsCapabilities,
      metrics: renderMetrics,
    });
    setBaselineCaptured(true);
  };

  const handleRenderMetrics = (metrics: RenderMetrics) => {
    setRenderMetrics(metrics);
    if (metrics.backend === 'WebGL2' || metrics.backend === 'WebGPU → WebGL2') {
      setGraphicsCapabilities((current) => current && !current.webgl2Available
        ? { ...current, webgl2Available: true, webglAvailable: true }
        : current);
    }
  };

  return (
    <div className="relative w-screen h-screen bg-[#07070a] overflow-hidden select-none">
      {inferredLighting?.analysis_warning && (
        <div role="status" className="absolute bottom-4 left-1/2 -translate-x-1/2 z-40 max-w-lg rounded-lg bg-amber-950/90 border border-amber-400/40 px-4 py-2 text-xs text-amber-100 pointer-events-none">
          Evidence note: {inferredLighting.analysis_warning}
        </div>
      )}
      {/* Top Floating Glass Toolbar */}
      {!clearView && <>
      <ViewerToolbar
        lightingMode={lightingMode}
        onSelectLighting={setLightingMode}
        showPointCloud={showPointCloud}
        onTogglePointCloud={() => setShowPointCloud((p) => !p)}
        graphicsQuality={graphicsQuality}
        resolvedGraphicsQuality={resolvedGraphicsQuality}
        onSelectGraphicsQuality={setGraphicsQuality}
        onResetCamera={handleResetCamera}
        onExit={onExit}
      />

      {/* Spatial HUD Overlay */}
      <SpatialHUD
        job={job}
        isPointerLocked={isPointerLocked}
        onLockPointer={handleLockPointer}
        onVirtualMove={(dx, dy) => setVirtualMove({ dx, dy })}
        onVirtualHeight={(delta) => {
          setVirtualHeightDelta(delta);
          setTimeout(() => setVirtualHeightDelta(0), 200);
        }}
        graphicsQuality={graphicsQuality}
        resolvedGraphicsQuality={resolvedGraphicsQuality}
        capabilities={graphicsCapabilities}
        renderMetrics={renderMetrics}
        baselineCaptured={baselineCaptured}
        onCaptureBaseline={handleCaptureBaseline}
      />
      </>}
      <button
        type="button"
        aria-pressed={clearView}
        onClick={() => setClearView((value) => !value)}
        className="absolute bottom-4 right-4 z-50 rounded-full border border-white/20 bg-black/70 px-4 py-2 text-xs text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400"
        title="Hide overlays to inspect fixtures behind the toolbar"
      >
        {clearView ? 'Show controls' : 'Clear view'}
      </button>

      {/* Universal WebGPU/WebGL Canvas Container */}
      <div
        ref={canvasContainerRef}
        onClick={handleLockPointer}
        className="w-full h-full cursor-crosshair"
      >
        {(graphicsCapabilities?.webgpuAvailable || graphicsCapabilities?.webglAvailable) && (
          <ViewerErrorBoundary key={`${glbUrl}-${cameraKey}`} onRetry={() => window.location.reload()}>
          <Canvas
            key={`${cameraKey}-${graphicsCapabilities.webgpuAvailable ? 'webgpu' : 'fallback'}`}
            shadows={graphicsProfile.shadows}
            dpr={graphicsProfile.dpr}
            performance={{ min: 0.65, max: 1, debounce: 350 }}
            camera={{ position: [0, 1.2, 0], fov: 68, near: 0.1, far: 50 }}
            gl={createRenderer as any}
          >
            <LightingRig
              mode={lightingMode}
              realTimeShadows={graphicsProfile.shadows}
              isSample={isSample}
              inferred={inferredLighting}
              objects={sceneObjects}
              roomBounds={roomBounds}
              shadowMapSize={graphicsProfile.shadowMapSize}
              maxDynamicLights={graphicsProfile.maxDynamicLights}
            />

            <Suspense fallback={null}>
              <RoomModel
                key={glbUrl}
                url={glbUrl}
                lightingMode={lightingMode}
                showPointCloud={showPointCloud}
                isSample={isSample}
                realTimeShadows={graphicsProfile.shadows}
                onLoaded={handleModelLoaded}
              />
            </Suspense>

            <PhotoLens calibration={!isSample ? inferredLighting?.projection_camera : undefined} />
            <FirstPersonControls
              obstacles={furnitureObstacles(sceneObjects)}
              initialPosition={!isSample ? inferredLighting?.projection_camera?.position ?? [0, 1.2, 0] : [0, 1.2, 0]}
              initialYaw={!isSample ? inferredLighting?.projection_camera?.yaw_degrees ?? 0 : 0}
              initialRotation={!isSample ? inferredLighting?.projection_camera?.rotation_matrix : undefined}
              roomBounds={roomBounds}
              roomLimits={roomLimits}
            />

            <MobileJoystickDriver
              obstacles={furnitureObstacles(sceneObjects)}
              virtualMove={virtualMove}
              virtualHeight={virtualHeightDelta}
              roomBounds={roomBounds}
              roomLimits={roomLimits}
            />

            <AdaptiveDpr pixelated={false} />
            {useUniversalRenderer && (
            <HighFidelityPipeline
                photoMode={!isSample}
                quality={resolvedGraphicsQuality}
                environmentIntensity={graphicsProfile.environmentIntensity}
              />
            )}
            <RenderDiagnostics modelLoadMs={modelLoadMs} onMetrics={handleRenderMetrics} />
          </Canvas>
          </ViewerErrorBoundary>
        )}
      </div>

      {graphicsCapabilities && !graphicsCapabilities.webgpuAvailable && !graphicsCapabilities.webglAvailable && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-[#07070a] px-8 text-center">
          <p className="max-w-lg text-sm text-slate-300">
            This browser could not create a WebGPU or WebGL renderer. Enable hardware acceleration or use a current browser.
          </p>
        </div>
      )}

      {/* Initial Loading Overlay */}
      {!modelLoaded && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#07070a]/90 backdrop-blur-md">
          <div className="relative w-16 h-16 mb-4 flex items-center justify-center">
            <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20 animate-ping" />
            <div className="w-12 h-12 rounded-full border-2 border-t-cyan-400 border-r-transparent border-b-cyan-500/30 border-l-transparent animate-spin" />
            <Box className="w-5 h-5 text-cyan-400 absolute" />
          </div>
          <span className="text-xs font-mono tracking-widest text-cyan-300 uppercase">
            Rendering Spatial 3D Scene...
          </span>
        </div>
      )}
    </div>
  );
};
