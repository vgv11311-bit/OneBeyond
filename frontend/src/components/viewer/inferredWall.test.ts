import {describe,it,expect} from 'vitest';
import {createInferredRearWall} from '../../../../scripts/splat-solid.js';
const fixture=()=>({center:[0,1,2],normal:[0,0,-1],width:4,height:3,max_offset:.5});
describe('explicit inferred rear completion',()=>{
 it('is disabled until requested and never projects a photograph',()=>{
  const wall=createInferredRearWall(fixture());
  expect(wall.mesh.visible).toBe(false);
  expect(wall.mesh.material.map).toBeNull();
  expect(wall.mesh.name).toContain('inferred');
  wall.setOffset(50);expect(wall.mesh.position.toArray()).toEqual([0,1,1.5]);
  wall.setOffset(NaN);expect(wall.mesh.position.toArray()).toEqual([0,1,1.5]);
 });
 it('rejects invalid placement rather than fabricating geometry',()=>{
  for(const patch of [{normal:[0,0,0]},{max_offset:Infinity},{width:-1},{center:[0,NaN,0]}])expect(()=>createInferredRearWall({...fixture(),...patch})).toThrow();
 });
});
