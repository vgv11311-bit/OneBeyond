import React, { useEffect, useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as THREE from 'three';
import { LightingMode } from '../../types';

interface RoomModelProps {
  url: string;
  lightingMode: LightingMode;
  showPointCloud: boolean;
  isSample?: boolean;
  realTimeShadows?: boolean;
  onLoaded?: () => void;
}

const prepareExternalAsset = (source: THREE.Group) => {
  const asset = source.clone(true);
  asset.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.castShadow = true;
      child.receiveShadow = true;
      child.frustumCulled = true;
      child.material = Array.isArray(child.material)
        ? child.material.map((material) => material.clone())
        : child.material.clone();
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        if (material instanceof THREE.MeshStandardMaterial) {
          material.envMapIntensity = 0.8;
          material.roughness = Math.max(0.3, Math.min(0.78, material.roughness));
          material.needsUpdate = true;
        }
      });
    }
  });
  return asset;
};

/** Curated 1K CC0 models loaded only by the sample, not every uploaded room. */
const SampleHeroAssets: React.FC = () => {
  const nightstand = useGLTF('/assets/polyhaven/painted_wooden_nightstand/painted_wooden_nightstand_1k.gltf');
  const clampLamp = useGLTF('/assets/polyhaven/desk_lamp_arm_01/desk_lamp_arm_01_1k.gltf');
  const nightstandScene = useMemo(() => prepareExternalAsset(nightstand.scene), [nightstand]);
  const clampLampScene = useMemo(() => prepareExternalAsset(clampLamp.scene), [clampLamp]);

  return (
    <>
      <primitive object={nightstandScene} position={[0.16, 0, -7.65]} rotation={[0, Math.PI, 0]} scale={0.92} />
      <primitive object={clampLampScene} position={[-2.55, 0.83, -3.15]} rotation={[0, Math.PI / 2, 0]} scale={0.72} />
    </>
  );
};

/** Lightweight, static contact shadows for the curated sample.
 * They preserve a grounded, sunlit look without asking a small browser GPU to
 * redraw the entire room into a shadow map every animation frame. */
const SampleFloorShadows: React.FC = () => {
  const shadows: Array<{ position: [number, number, number]; scale: [number, number, number]; opacity: number }> = [
    { position: [-1.4, 0.012, -7.65], scale: [1.65, 0.86, 1], opacity: 0.22 },
    { position: [-2.55, 0.012, -3.15], scale: [1.08, 0.58, 1], opacity: 0.14 },
    { position: [1.25, 0.012, -5.9], scale: [0.34, 0.22, 1], opacity: 0.2 },
    { position: [3.15, 0.012, -5.65], scale: [0.42, 0.28, 1], opacity: 0.16 },
    { position: [0.16, 0.012, -7.65], scale: [0.46, 0.28, 1], opacity: 0.16 },
  ];

  return (
    <>
      {shadows.map((shadow, index) => (
        <mesh key={index} position={shadow.position} rotation={[-Math.PI / 2, 0, 0]} scale={shadow.scale} renderOrder={1}>
          <circleGeometry args={[1, 32]} />
          <meshBasicMaterial color="#2a1b10" transparent opacity={shadow.opacity} depthWrite={false} />
        </mesh>
      ))}
    </>
  );
};

export const RoomModel: React.FC<RoomModelProps> = ({
  url,
  lightingMode,
  showPointCloud,
  isSample = false,
  realTimeShadows = false,
  onLoaded,
}) => {
  const gltf = useLoader(GLTFLoader, url);

  const processedScene = useMemo(() => {
    const scene = gltf.scene.clone(true);

    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        // Each reconstructed object gets its own material instance, so PBR
        // tuning for fabric, wood or metal cannot leak into another object.
        child.material = Array.isArray(child.material)
          ? child.material.map((material) => material.clone())
          : child.material.clone();
        // The enclosing room is only a shadow receiver. Rendering every wall
        // and floor again into the shadow map roughly doubles frame work and
        // was the source of the slow walk through the sample.
        const isFloor = ['Oak_Floor', 'Floor'].includes(child.name);
        const isRoomShell = [
          'Oak_Floor', 'Floor', 'Ceiling', 'Back_Wall', 'Left_Wall',
          'Right_Wall', 'Front_Wall', 'Accent_Wall',
        ].includes(child.name) || /^(Back|Left|Right)_Wall_Panel_\d+$/.test(child.name);
        child.castShadow = realTimeShadows && !isRoomShell;
          // The shell does not cast, so receivers can include walls and all
          // furniture without inward room-box self-shadow artifacts.
          child.receiveShadow = realTimeShadows && (isFloor || !isSample);

        const materials = Array.isArray(child.material) ? child.material : [child.material];
          const semanticName = `${child.name} ${materials.map((material) => material.name).join(' ')}`.toLowerCase();
          if (/glass/.test(semanticName)) child.castShadow = false;
        materials.forEach((material) => {
          // The camera sits inside the room boxes, so their inward faces need
          // double-sided lighting. Furniture stays front-facing for speed.
          material.side = isRoomShell || /observed.exterior/.test(semanticName) ? THREE.DoubleSide : THREE.FrontSide;
          // The GLB now contains only neutral, independently navigable meshes.
          // Keep depth writing on, so room walls and furniture occlude correctly
          // from every WASD camera angle.
          // Uploaded GLBs own their alpha/PBR policy. In particular, an
          // "opaque window glass" must not become a transparent opening.
          if (isSample) {
            const isGlassSurface = /glass/.test(semanticName);
            material.depthWrite = !isGlassSurface;
            material.transparent = isGlassSurface;
            material.opacity = isGlassSurface ? Math.min(material.opacity, 0.42) : 1;
          }
          // Older exports can contain a paint layer directly behind an
          // observed wallpaper finish. Keep the photo-evidenced wallpaper in
          // front while those saved rooms are being rebuilt; new exports cut
          // overlapping finishes out of the underlying paint mesh.
          if (/observed_.*_wallpaper/.test(semanticName)) {
            material.polygonOffset = true;
            material.polygonOffsetFactor = -2;
            material.polygonOffsetUnits = -2;
          }
          if (material instanceof THREE.MeshStandardMaterial) {
            material.envMapIntensity = isSample ? 0.75 : 0.2;

            if (!isSample) {
              // Do not reinterpret exported materials by object name. A
              // sofa frame is not necessarily wood and a table can be stone.
            } else if (isRoomShell) {
              // Preserve the exported floor finish (e.g. carpet vs tile).
              if (!semanticName.includes('floor')) material.roughness = 0.96;
              material.metalness = 0;
            } else if (/(screen|glass)/.test(semanticName)) {
              material.roughness = 0.16;
              material.metalness = 0.12;
            } else if (/(metal|chrome|steel|fixture|brushed)/.test(semanticName)) {
              material.roughness = 0.32;
              material.metalness = Math.max(0.55, material.metalness);
            } else if (/(wood|table|desk|door|nightstand|cabinet|frame)/.test(semanticName)) {
              material.roughness = 0.58;
              material.metalness = 0.02;
            } else if (/(fabric|cloth|rug|curtain|bed|pillow|mattress|sofa|chair|upholstery)/.test(semanticName)) {
              material.roughness = 0.84;
              material.metalness = 0;
              // Rough photo fabrics should not pick up a strong cool studio
              // reflection that competes with their local fixture lighting.
              if (!isSample) material.envMapIntensity = 0.15;
            } else {
              material.roughness = Math.max(0.35, Math.min(0.9, material.roughness));
            }

            if (isSample && /(screen|lamp|bulb)/.test(semanticName) && material.emissiveIntensity <= 1 && material.emissive.getHex() === 0) {
              material.emissive.copy(material.color).multiplyScalar(0.12);
              material.emissiveIntensity = 0.2;
            }
          }
          material.needsUpdate = true;
        });
        child.renderOrder = 0;

        if (lightingMode === 'wireframe') {
          materials.forEach((material) => material.dispose());
          child.material = new THREE.MeshBasicMaterial({
            color: 0x00f0ff,
            wireframe: true,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide,
          });
        }
      }

      if (child instanceof THREE.Points) {
        child.visible = showPointCloud;
        const clones = (Array.isArray(child.material) ? child.material : [child.material]).map((material) => material.clone());
        child.material = Array.isArray(child.material) ? clones : clones[0];
        clones.forEach((material) => {
          if (material instanceof THREE.PointsMaterial) {
            material.size = 0.045;
            material.sizeAttenuation = true;
            material.transparent = true;
            material.opacity = 0.95;
          }
        });
      }
    });

    // Treat reconstruction transforms as immutable render data. Lighting,
    // diagnostics and quality updates can change appearance, never placement.
    scene.traverse((child) => {
      child.updateMatrix();
      child.matrixAutoUpdate = false;
    });

    return scene;
  }, [gltf, lightingMode, showPointCloud, isSample, realTimeShadows]);

  useEffect(() => () => {
    // Only dispose owned material clones, never loader-shared textures or
    // geometry. Quality toggles otherwise leave an entire material set alive.
    processedScene.traverse((child) => {
      if (child instanceof THREE.Mesh || child instanceof THREE.Points) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((material) => material.dispose());
      }
    });
  }, [processedScene]);

  useEffect(() => {
    if (onLoaded) onLoaded();
  }, [onLoaded]);

  useEffect(() => () => {
    // A reconstruction URL includes a model revision, but explicitly evicting
    // its loader entry on unmount also prevents a revisited job from retaining
    // stale GPU scene data after the GLB was rebuilt.
    useLoader.clear(GLTFLoader, url);
  }, [url]);

  return (
    <>
      <primitive object={processedScene} />
      {isSample && <>
        <SampleHeroAssets />
      </>}
    </>
  );
};
