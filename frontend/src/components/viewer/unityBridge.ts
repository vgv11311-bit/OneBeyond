import { SpaceJob } from '../../types';

export type UnityBuild = { version: number; loaderUrl: string; dataUrl: string; frameworkUrl: string; codeUrl: string; buildId: string };
export type UnityInstance = { SendMessage: (object: string, method: string, value: string) => void; Quit: () => Promise<void> };
export type UnityEvent = { type: string; session: string; message?: string; quality?: string; fps?: number; frameMs?: number };
declare global {
  interface Window { createUnityInstance?: (canvas: HTMLCanvasElement, config: Record<string, unknown>, progress: (value:number)=>void) => Promise<UnityInstance>; }
}

export function unityBuildConfig(value: unknown, origin: string) {
  const manifest=value as UnityBuild;
  if (manifest?.version!==1 || !/^[0-9]{14}$/.test(manifest.buildId)) throw new Error('Unity build manifest is invalid.');
  const files: Record<string,string>={};
  for (const field of ['loaderUrl','dataUrl','frameworkUrl','codeUrl'] as const) {
    if (typeof manifest[field]!=='string' || !/^Build\/[a-zA-Z0-9_.-]+$/.test(manifest[field])) throw new Error('Unsafe Unity build path.');
    files[field]=new URL(`/unity/${manifest[field]}?v=${manifest.buildId}`,origin).href;
  }
  return files;
}
const vector=(v:unknown,fallback:number[]):number[] => Array.isArray(v)&&v.length===3&&v.every(Number.isFinite)?v:fallback;
const mirror=(v:number[])=>[-v[0],v[1],v[2]];
const dimension=(v:number|undefined,fallback:number)=>typeof v==='number'&&Number.isFinite(v)&&v>0?v:fallback;
const finite=(v:unknown,fallback=0)=>typeof v==='number'&&Number.isFinite(v)?v:fallback;

/** glTFast mirrors X, not Z. Preserve the exported world frame and lens. */
export function unitySceneRequest(job: SpaceJob, session:string, origin:string) {
  const lighting=job.metrics?.extra_details?.lighting;
  const calibration=lighting?.projection_camera;
  const r=calibration?.rotation_matrix;
  const validRotation=Array.isArray(r)&&r.length===3&&r.every(row=>Array.isArray(row)&&row.length===3&&row.every(Number.isFinite));
  const yaw=finite(calibration?.yaw_degrees)*Math.PI/180;
  const forward=validRotation?[-r[0][2],-r[1][2],-r[2][2]]:[-Math.sin(yaw),0,-Math.cos(yaw)];
  const up=validRotation?[r[0][1],r[1][1],r[2][1]]:[0,1,0];
  const url=new URL(job.glb_url||`/api/spaces/${encodeURIComponent(job.job_id)}/model.glb`,origin);
  if(url.origin!==new URL(origin).origin || !/^https?:$/.test(url.protocol)) throw new Error('Room models must be served from this application.');
  const rawObjects=job.metrics?.extra_details?.scene_objects;
  const objects=(Array.isArray(rawObjects)?rawObjects:[]).filter(o=>o&&typeof o==='object') as Array<Record<string,any>>;
  return {session,url:url.href,
    width:dimension(job.metrics?.estimated_room_width,6),length:dimension(job.metrics?.estimated_room_length,7),height:dimension(job.metrics?.estimated_room_height,3),
    position:mirror(vector(calibration?.position,[0,1.2,0])), forward:mirror(forward), up:mirror(up),
    fy:dimension(calibration?.intrinsics?.[1]?.[1],0),photoAspect:dimension(calibration?.source_shape?.[1],1)/dimension(calibration?.source_shape?.[0],1),
    ambient:dimension(lighting?.ambient_intensity,.5),exposure:dimension(lighting?.exposure,1),
    obstacles:objects.filter(o=>/^(sofa|bed|cabinet|wardrobe|desk|table|coffee_table|nightstand|side_table|chair)$/.test(o.label)
      && Array.isArray(o.center)&&o.center.length===3&&o.center.every(Number.isFinite)
      && Array.isArray(o.size)&&o.size.length===3&&o.size.every((n:number)=>Number.isFinite(n)&&n>0))
      .map(o=>({center:mirror(o.center),size:o.size,yaw:-finite(o.yaw_degrees)})),
    fixtures:objects.filter(o=>/lamp|light|chandelier/.test(String(o.label))&&o.fixture_light&&Array.isArray(o.center))
      .slice(0,8).map(o=>({position:mirror(vector(o.center,[0,2,0])),color:vector(o.fixture_light.color,[1,1,1]),
        intensity:dimension(o.fixture_light.intensity,1),ceiling:/ceiling/.test(o.label)})),
  };
}

let loader: Promise<void>|undefined;
let loaderAddress='';
export function loadUnityLoader(url:string) {
  if(loader && loaderAddress===url) return loader;
  loaderAddress=url;
  loader=new Promise<void>((resolve,reject)=>{
    const script=document.createElement('script'); script.src=url;script.async=true;
    script.onload=()=>window.createUnityInstance?resolve():reject(new Error('Unity loader did not initialize.'));
    script.onerror=()=>{loader=undefined;script.remove();reject(new Error('Unity player download failed.'));};
    document.head.appendChild(script);
  });
  return loader;
}

// Serialize entire player lifetimes, including pending startup, so React
// StrictMode and rapid room changes never allocate overlapping Unity heaps.
let previousPlayer=Promise.resolve();
export function unityPlayerLease() {
  const wait=previousPlayer.catch(()=>undefined);
  let release!:()=>void;
  previousPlayer=new Promise<void>(resolve=>{release=resolve;});
  return {wait,release};
}
