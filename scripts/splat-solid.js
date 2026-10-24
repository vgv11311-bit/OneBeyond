// Solid completion underneath the photographic layer. No texture/color edits
// to splats; hidden geometry and collision proxies remain inferred estimates.
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

// Inferred architectural completion; no pasted whole-photo objects or API.
export function createInferredRearWall(definition){
 const vector=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
 if(!vector(definition.center)||!vector(definition.normal)||!Number.isFinite(definition.width)||!Number.isFinite(definition.height)||definition.width<=0||definition.height<=0||!Number.isFinite(definition.max_offset)||definition.max_offset<=0)throw new Error('Invalid inferred rear wall');
 const normal=new THREE.Vector3(...definition.normal).normalize();
 if(normal.lengthSq()<.99)throw new Error('Invalid inferred wall normal');
 const mesh=new THREE.Mesh(new THREE.PlaneGeometry(definition.width,definition.height),new THREE.MeshBasicMaterial({color:0x939393,side:THREE.DoubleSide}));
 const center=new THREE.Vector3(...definition.center);mesh.position.copy(center);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),normal);
 mesh.name='Explicitly_inferred_neutral_rear_wall';mesh.visible=definition.enabled===true;
 return {mesh,setOffset:offset=>{if(Number.isFinite(offset))mesh.position.copy(center).addScaledVector(normal,THREE.MathUtils.clamp(offset,-definition.max_offset,definition.max_offset));}};
}

export function createEnclosure(navigation){
 const scene=new THREE.Scene(),low=navigation.bounds.min,high=navigation.bounds.max;
 const size=new THREE.Vector3(...high).sub(new THREE.Vector3(...low));
 const center=new THREE.Vector3(...low).addScaledVector(size,.5);
 const rectangle=navigation.enclosure_rect;
 if(rectangle){
  size.x=rectangle.max[0]-rectangle.min[0];size.z=rectangle.max[1]-rectangle.min[1];
  const x=(rectangle.min[0]+rectangle.max[0])*.5,z=(rectangle.min[1]+rectangle.max[1])*.5;
  center.x=rectangle.center[0]+Math.cos(rectangle.yaw)*x+Math.sin(rectangle.yaw)*z;
  center.z=rectangle.center[1]-Math.sin(rectangle.yaw)*x+Math.cos(rectangle.yaw)*z;
 }
 const appearance=navigation.appearance;
 const base=new THREE.Color(appearance?.wall_color||navigation.wall_color||'#aaa399');
 // Unobserved architecture is always a simple solid completion. A sampled
 // height palette can include furniture shadows and creates false black bands.
 const wall=new THREE.MeshBasicMaterial({color:base,side:THREE.BackSide});
 const ceiling=new THREE.MeshBasicMaterial({color:base,side:THREE.BackSide});
 const floor=new THREE.MeshBasicMaterial({color:base.clone().multiplyScalar(.85),side:THREE.BackSide});
 const box=new THREE.Mesh(new THREE.BoxGeometry(size.x,size.y,size.z),[wall,wall,ceiling,floor,wall,wall]);
 box.position.copy(center);if(rectangle)box.rotation.y=rectangle.yaw;scene.add(box);return scene;
}

export function constrainMove(from,desired,solid){
 const low=solid.bounds.min,high=solid.bounds.max,radius=.16;
 const next=desired.clone();
 next.x=THREE.MathUtils.clamp(next.x,low[0]+radius,high[0]-radius);
 next.z=THREE.MathUtils.clamp(next.z,low[2]+radius,high[2]-radius);
 next.y=THREE.MathUtils.clamp(desired.y,low[1]+.5,high[1]-.2);
 const blocked=(point,previous)=>solid.obstacles.some(object=>{
  const [x,y,z]=object.center,[w,d,h]=object.size;
  if(y+h/2 < point.y-1.05 || y-h/2 > point.y+.15)return false;
  const yaw=THREE.MathUtils.degToRad(object.yaw_degrees||0),c=Math.cos(yaw),s=Math.sin(yaw);
  const local=p=>[c*(p.x-x)-s*(p.z-z),s*(p.x-x)+c*(p.z-z)];
  const [a,b]=local(point),[pa,pb]=local(previous);
  const inside=Math.abs(a)<w/2+radius&&Math.abs(b)<d/2+radius;
  const wasInside=Math.abs(pa)<w/2+radius&&Math.abs(pb)<d/2+radius;
  // Allow escape if an estimated box overlaps a captured camera; never trap it.
  return inside&&(!wasInside || a*a+b*b < pa*pa+pb*pb-1e-9);
 });
 const slide=from.clone();slide.x=next.x;
 if(blocked(slide,from))slide.x=from.x;
 const prior=slide.clone();slide.z=next.z;
 if(blocked(slide,prior))slide.z=prior.z;
 slide.y=next.y;return slide;
}

export async function createSolidRoom(config){
 if(!config.solid?.enabled)return null;
 const solid=config.solid;
 const scene=new THREE.Scene();
 const [low,high]=[solid.bounds.min,solid.bounds.max];
 const center=new THREE.Vector3(...low).add(new THREE.Vector3(...high)).multiplyScalar(.5);
 const size=new THREE.Vector3(...high).sub(new THREE.Vector3(...low));
 const color=(key,fallback)=>{
  const rgb=solid.colors?.[key];
  if(!Array.isArray(rgb)||rgb.length!==3||!rgb.every(Number.isFinite))return new THREE.Color(fallback);
  // Analysis stores display RGB in 0..255, not linear-light channel values.
  const denominator=Math.max(...rgb)>1?255:1;
  return new THREE.Color().setRGB(...rgb.map(v=>THREE.MathUtils.clamp(v/denominator,0,1)),THREE.SRGBColorSpace);
 };
 scene.background=color('wall',0xb9b0a0);
 scene.add(new THREE.HemisphereLight(0xfffaf0,0x77716b,2));
 const light=new THREE.DirectionalLight(0xfff8ee,1.2);light.position.set(center.x,high[1],center.z+size.z*.2);scene.add(light);
 const gltf=await new GLTFLoader().loadAsync(new URL(solid.glb_asset,location.href).href);
 scene.add(gltf.scene);
 let wallMaterial=new THREE.MeshStandardMaterial({color:color('wall',0xb9b0a0),roughness:.9,side:THREE.DoubleSide});
 let doorMaterial=new THREE.MeshStandardMaterial({color:0x744627,roughness:.65});
 const entryNodes=[];
 gltf.scene.traverse(node=>{
  if(!node.isMesh)return;
  if(node.name==='Front_Wall'&&node.material?.clone)wallMaterial=node.material.clone();
  if(node.name==='Entry_Door'&&node.material?.clone)doorMaterial=node.material.clone();
  // Move the inferred camera-side enclosure beyond *all* registered cameras,
  // rather than leaving the original single-anchor wall across other views.
  if(node.name==='Front_Wall')node.visible=false;
  if(node.name.startsWith('Entry_Door'))entryNodes.push(node);
 });
 gltf.scene.updateMatrixWorld(true);
 // Reuse the generated room's bevelled door, frame, handle and materials when
 // available. Only relocate the inferred assembly, never retexture the room.
 for(const node of entryNodes){
  const point=node.getWorldPosition(new THREE.Vector3());point.x+=center.x;point.z+=high[2]-.15;
  node.position.copy(node.parent.worldToLocal(point));node.updateMatrix();
 }
 // Keep the inferred door in the enclosure, independently of editable furniture.
 for(const node of entryNodes)scene.attach(node);
 const box=(name,dimensions,position,material)=>{
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(...dimensions),material);mesh.name=name;mesh.position.set(...position);scene.add(mesh);return mesh;
 };
 const floorMaterial=new THREE.MeshStandardMaterial({color:color('floor',0xb4ad9e),roughness:.9});
 const ceilingMaterial=new THREE.MeshStandardMaterial({color:color('ceiling',0xd3cabb),roughness:1});
 // Six opaque backing surfaces also seal window openings and small GLB cracks.
 box('Completion_Floor',[size.x+.2,.08,size.z+.2],[center.x,low[1]-.04,center.z],floorMaterial);
 box('Completion_Ceiling',[size.x+.2,.08,size.z+.2],[center.x,high[1]+.04,center.z],ceilingMaterial);
 box('Completion_Left',[.08,size.y,size.z+.2],[low[0]-.04,center.y,center.z],wallMaterial);
 box('Completion_Right',[.08,size.y,size.z+.2],[high[0]+.04,center.y,center.z],wallMaterial);
 box('Completion_Back',[size.x+.2,size.y,.08],[center.x,center.y,low[2]-.04],wallMaterial);
 box('Inferred_CameraSide_Wall',[size.x+.2,size.y,.08],[center.x,center.y,high[2]+.04],wallMaterial);
 const dw=Math.min(.95,Math.max(.72,size.x*.22)),dh=Math.min(2.15,Math.max(1.9,size.y*.78)),dz=high[2]-.022;
 if(!entryNodes.length){
 box('Inferred_Door',[dw,dh,.04],[center.x,low[1]+dh/2,dz],doorMaterial);
 const trim=new THREE.MeshStandardMaterial({color:0x483321,roughness:.6});
 for(const sign of [-1,1])box('Inferred_Door_Frame',[.055,dh+.055,.06],[center.x+sign*(dw/2+.0275),low[1]+dh/2,dz-.025],trim);
 box('Inferred_Door_Frame_Top',[dw+.11,.055,.06],[center.x,low[1]+dh+.0275,dz-.025],trim);
 const handle=new THREE.Mesh(new THREE.SphereGeometry(.025,12,8),new THREE.MeshStandardMaterial({color:0xb8b1a0,metalness:.8,roughness:.25}));
 handle.position.set(center.x+dw*.34,low[1]+dh*.5,dz-.055);scene.add(handle);
 }
 const raycaster=new THREE.Raycaster();
 // Do not swap photographed furniture for a different room when walking.
 return {scene,photoWeight:()=>1,setPhotographic:enabled=>{gltf.scene.visible=!enabled;},constrain:(from,to)=>constrainMove(from,to,solid),
  inspect:()=>{scene.updateMatrixWorld(true);raycaster.set(center,new THREE.Vector3(0,0,1));return {opaqueEnvelopeSurfaces:6,inferredDoor:true,roomBounds:solid.bounds,colorsLinear:{floor:floorMaterial.color.toArray(),ceiling:ceilingMaterial.color.toArray()},backingHit:raycaster.intersectObjects(scene.children,true).length>0};}};
}
