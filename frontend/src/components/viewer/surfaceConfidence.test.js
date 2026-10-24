import {describe, it, expect} from 'vitest';
import {surfaceWeight} from '../../../../scripts/splat-surface-confidence.js';

const anchor={name:'a',position:[0,0,0],rotation:[[1,0,0],[0,1,0],[0,0,1]]};
const cameras=[anchor,{...anchor,name:'b',position:[1,0,0]}];
describe('single-photo depth support',()=>{
 it('retains the observed surface at its source capture',()=>{
  expect(surfaceWeight([0,0,0],[0,0,-1],anchor,cameras)).toBe(1);
 });
 it('does not project that photo over another captured viewpoint',()=>{
  expect(surfaceWeight([1,0,0],[0,0,-1],anchor,cameras)).toBe(0);
 });
 it('fades continuously between captures without depending on metric scale',()=>{
  const a=surfaceWeight([.2,0,0],[0,0,-1],anchor,cameras);
  const b=surfaceWeight([.3,0,0],[0,0,-1],anchor,cameras);
  expect(a).toBeGreaterThan(b);expect(b).toBeGreaterThan(0);
  const scaled=cameras.map(c=>({...c,position:c.position.map(v=>v*100)}));
  expect(surfaceWeight([20,0,0],[0,0,-1],scaled[0],scaled)).toBeCloseTo(a);
 });
 it('removes the patch when looking away or far outside its support',()=>{
  expect(surfaceWeight([0,0,0],[0,0,1],anchor,cameras)).toBe(0);
  expect(surfaceWeight([0,0,100],[0,0,-1],anchor,cameras)).toBeLessThan(.001);
 });
});
