import {describe,it,expect} from 'vitest';
import {createNavigation,selectSurface,WALK_SETTINGS,photographicWalkSpeed} from '../../../../scripts/splat-navigation.js';

const fixture=()=>({bounds:{min:[0,-2,0],max:[10,3,10]},radius:.2,cell:.2,width:51,depth:51,
 occupied:Array.from({length:51},(_,z)=>z*51+25)});
describe('photographic collision movement',()=>{
 it('recovers a blocked spawn without erasing furniture or changing elevation',()=>{
  const data={...fixture(),walking_plane:'level_floor'};
  const nav=createNavigation(data),before=nav.inspect().occupiedCells;
  const result=nav.constrain([5.05,.8,3],[4,.8,3]);
  expect(nav.blocked(result)).toBe(false);expect(result[1]).toBe(.8);
  expect(nav.inspect().occupiedCells).toBe(before);
  expect(nav.blocked([5.05,.8,3])).toBe(true);
 });
 it('rejects malformed collision metadata and blocks invalid or exterior probes',()=>{
  for(const patch of [{radius:Infinity},{width:1.5},{occupied:[-1]},{occupied:[2601]},{bounds:{min:[0,0],max:[1,1]}}])expect(()=>createNavigation({...fixture(),...patch})).toThrow();
  expect(()=>createNavigation({...fixture(),enclosure_rect:{center:[0,0],yaw:NaN,min:[-1,-1],max:[1,1]}})).toThrow();
  const nav=createNavigation(fixture());
  expect(nav.blocked([-1,0,2])).toBe(true);expect(nav.blocked([NaN,0,2])).toBe(true);
  expect(()=>nav.constrain([NaN,0,0],[1,0,1])).toThrow();
 });
 it('uses the legacy walk tuning with explicit estimated-scale conversion',()=>{
  expect(WALK_SETTINGS).toMatchObject({speed:4,look:.0022,damping:15,sprint:1.8});
  expect(photographicWalkSpeed(1.4)).toBe(4);expect(photographicWalkSpeed(2.8)).toBe(8);
 });
 it('contains a rotated rectangular enclosure',()=>{
  const data={...fixture(),occupied:[],enclosure_rect:{center:[5,5],yaw:Math.PI/4,min:[-2,-2],max:[2,2]}};
  const nav=createNavigation(data);expect(nav.blocked([5,0,5])).toBe(false);
  expect(nav.blocked([8,0,8])).toBe(true);
  expect(nav.blocked(nav.constrain([5,0,5],[9,0,9]))).toBe(false);
 });
 it('stops a long step at an observed wall, with radius clearance',()=>{
  const nav=createNavigation(fixture());
  const p=nav.constrain([2,0,3],[9,0,3]);
  expect(p[0]).toBeLessThan(4.81);expect(p[0]).toBeGreaterThan(4.6);
 });
 it('slides along a wall rather than tunnelling diagonally',()=>{
  const nav=createNavigation(fixture()),p=nav.constrain([2,0,2],[9,0,8]);
  expect(p[0]).toBeLessThan(4.81);expect(p[2]).toBeCloseTo(8);
 });
 it('keeps all boundary exits closed and does not allow flying over furniture',()=>{
  const nav=createNavigation({...fixture(),occupied:[]});
  const low=nav.constrain([2,0,2],[-20,99,-20]);
  expect(low[0]).toBeCloseTo(.2);expect(low[2]).toBeCloseTo(.2);expect(low[1]).toBe(0);
  const p=nav.constrain([2,0,2],[20,-99,20]);
  expect(p[0]).toBeCloseTo(9.8);expect(p[2]).toBeCloseTo(9.8);expect(p[1]).toBe(0);
 });
 it('blocks a sofa footprint even when the camera looks over it',()=>{
  const data=fixture();data.occupied=[];
  for(let z=20;z<30;z++)for(let x=20;x<30;x++)data.occupied.push(z*51+x);
  const nav=createNavigation(data),p=nav.constrain([5,0,2],[5,0,8]);
  expect(p[2]).toBeLessThan(3.81);expect(nav.blocked(p)).toBe(false);
 });
 it('allows explicit bounded camera elevation without bypassing a furniture footprint',()=>{
  const nav=createNavigation(fixture());
  expect(nav.constrain([2,0,2],[2,99,2],{allowElevation:true})[1]).toBeCloseTo(2.8);
  expect(nav.constrain([2,0,2],[2,-99,2],{allowElevation:true})[1]).toBeCloseTo(-1.8);
  const blocked=nav.constrain([2,0,2],[9,2,2],{allowElevation:true});
  expect(blocked[0]).toBeLessThan(4.81);expect(nav.blocked(blocked)).toBe(false);
 });
 it('is invariant under scene scale and rejects nonfinite movement',()=>{
  const data=fixture(),p=createNavigation(data).constrain([2,0,2],[9,0,8]);
  const scaled={...data,radius:2,cell:2,bounds:{min:[0,-20,0],max:[100,30,100]}};
  const q=createNavigation(scaled).constrain([20,0,20],[90,0,80]);
  q.forEach((v,i)=>expect(v/10).toBeCloseTo(p[i]));
  expect(createNavigation(data).constrain([2,0,2],[NaN,0,0])).toEqual([2,0,2]);
 });
});
describe('multi-view photographic support',()=>{
 const cameras=[0,5,10].map(x=>({position:[x,0,0],rotation:[[1,0,0],[0,1,0],[0,0,1]]}));
 it('keeps photo geometry available while walking, choosing nearby captured support',()=>{
  expect(selectSurface([4,0,0],[0,0,-1],cameras)).toBe(1);
  expect(selectSurface([9,0,0],[0,0,-1],cameras,1)).toBe(2);
 });
 it('does not flicker across a nearest-view boundary',()=>{
  expect(selectSurface([2.51,0,0],[0,0,-1],cameras,0)).toBe(0);
 });
});
