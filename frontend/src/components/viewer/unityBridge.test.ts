import { describe,it,expect } from 'vitest';
import { unityBuildConfig,unitySceneRequest,unityPlayerLease } from './unityBridge';
import { SpaceJob } from '../../types';
const job={job_id:'any room',name:'Test',created_at:'',status:'completed',input_type:'photos',files_count:1,progress:100,stage:'done',logs:[],metrics:{total_frames_processed:1,keypoints_detected:0,points_3d_triangulated:0,processing_duration_sec:0,engine_name:'test',estimated_room_width:4,estimated_room_length:5,estimated_room_height:3,
  extra_details:{lighting:{projection_camera:{position:[1,1.4,-.2],rotation_matrix:[[1,0,0],[0,1,0],[0,0,1]],intrinsics:[[1,0,0],[0,.8,0],[0,0,1]],source_shape:[100,200]}},
    scene_objects:[{label:'sofa',center:[1,.5,-3],size:[2,1,1],yaw_degrees:90}]}}} as SpaceJob;
describe('Unity viewer contract',()=>{
  it('preserves scale and mirrors the camera and collision frame consistently',()=>{
    const p=unitySceneRequest(job,'session','http://localhost:5000');
    expect(p.position).toEqual([-1,1.4,-.2]);expect(p.forward.map(v=>v===0?0:v)).toEqual([0,0,-1]);
    expect(p.width).toBe(4);expect(p.fy).toBe(.8);expect(p.photoAspect).toBe(2);
    expect(p.obstacles[0]).toEqual({center:[-1,.5,-3],size:[2,1,1],yaw:-90});
    expect(p.url).toContain('any%20room');
  });
  it('rejects off-origin models and malformed manifests',()=>{
    expect(()=>unitySceneRequest({...job,glb_url:'https://elsewhere/model.glb'},'s','http://localhost')).toThrow();
    expect(()=>unityBuildConfig({version:1,buildId:'20261017000000',loaderUrl:'../evil.js'},'http://localhost')).toThrow();
  });
  it('accepts a versioned same-origin release build',()=>{
    const result=unityBuildConfig({version:1,buildId:'20261017000000',loaderUrl:'Build/a.loader.js',dataUrl:'Build/a.data.unityweb',frameworkUrl:'Build/a.framework.js.unityweb',codeUrl:'Build/a.wasm.unityweb'},'http://localhost');
    expect(result.codeUrl).toBe('http://localhost/unity/Build/a.wasm.unityweb?v=20261017000000');
  });
  it('serializes player disposal before the next startup',async()=>{
    const first=unityPlayerLease();const second=unityPlayerLease();await first.wait;
    let ready=false;const waiting=second.wait.then(()=>{ready=true;});await Promise.resolve();expect(ready).toBe(false);
    first.release();await waiting;expect(ready).toBe(true);second.release();
  });
  it('uses safe defaults for missing metrics without inventing objects',()=>{
    const p=unitySceneRequest({...job,metrics:undefined},'s','http://localhost');
    expect(p.width).toBe(6);expect(p.length).toBe(7);expect(p.obstacles).toEqual([]);expect(p.fixtures).toEqual([]);
    expect(p.position).toEqual([-0,1.2,0]);expect(p.fy).toBe(0);
  });
});
