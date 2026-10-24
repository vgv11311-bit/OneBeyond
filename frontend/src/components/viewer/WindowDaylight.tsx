import React, { useMemo } from 'react';
import * as THREE from 'three';
import { InferredLighting } from '../../types';

type WindowEvidence = NonNullable<InferredLighting['daylight']>['windows'][number];

export const WindowDaylight: React.FC<{
  window: WindowEvidence; shadows: boolean; mapSize: number; room: [number,number,number];
}> = ({ window: source, shadows, mapSize, room }) => {
  const rig = useMemo(() => {
    const center = new THREE.Vector3(...source.center);
    const inward = new THREE.Vector3(source.wall === 'left' ? 1 : source.wall === 'right' ? -1 : 0,0,source.wall === 'back' ? 1 : 0);
    const target = new THREE.Object3D();
    target.position.copy(source.target ? new THREE.Vector3(...source.target) : center.clone().addScaledVector(inward, room[0]*.65).setY(.4));
    const direction = center.clone().sub(target.position).normalize();
    const position = center.clone().addScaledVector(direction, Math.max(...room)*2);
    const extent = Math.max(...room)*4;
    const shape = new THREE.Shape();
    shape.moveTo(-extent,-extent); shape.lineTo(extent,-extent);
    shape.lineTo(extent,extent); shape.lineTo(-extent,extent); shape.closePath();
    const hole = new THREE.Path();
    const halfW = Math.max(.05,source.size[0]/2), halfH = Math.max(.05,source.size[2]/2);
    hole.moveTo(-halfW,-halfH); hole.lineTo(-halfW,halfH);
    hole.lineTo(halfW,halfH); hole.lineTo(halfW,-halfH); hole.closePath();
    shape.holes.push(hole);
    return { target, position, center, inward, aperture: new THREE.ShapeGeometry(shape) };
  }, [source, room]);
  React.useEffect(() => () => rig.aperture.dispose(), [rig]);
  const direct = source.direct_intensity > 0 && shadows;
  return <>
    <primitive object={rig.target} />
    <spotLight position={rig.center.clone().addScaledVector(rig.inward,.12)} target={rig.target}
      color={new THREE.Color().setRGB(...source.color,THREE.SRGBColorSpace)}
      intensity={source.diffuse_intensity * 8} angle={1.25} penumbra={1}
      distance={Math.max(...room)*2} decay={2} />
    {direct && <>
      <directionalLight position={rig.position} target={rig.target}
        color={new THREE.Color().setRGB(...source.color,THREE.SRGBColorSpace)}
        intensity={source.direct_intensity} castShadow
        shadow-mapSize-width={mapSize} shadow-mapSize-height={mapSize}
        shadow-camera-left={-Math.max(...room)} shadow-camera-right={Math.max(...room)}
        shadow-camera-top={Math.max(...room)} shadow-camera-bottom={-Math.max(...room)}
        shadow-camera-near={.1} shadow-camera-far={Math.max(...room)*6}
        shadow-bias={-.00015} shadow-normalBias={.006} />
      {/* Shadow-only wall with the measured window opening. It prevents
          directional sunlight illuminating surfaces outside the aperture. */}
      <mesh geometry={rig.aperture} position={rig.center}
        rotation={[0,source.wall === 'left' || source.wall === 'right' ? Math.PI/2 : 0,0]} castShadow>
        <meshBasicMaterial side={THREE.DoubleSide} colorWrite={false} depthWrite={false} />
      </mesh>
    </>}
  </>;
};
