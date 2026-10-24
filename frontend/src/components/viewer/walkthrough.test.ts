import { describe, it, expect } from 'vitest';
import { furnitureObstacles, moveWithinRoom, Obstacle } from './walkthrough';
const limits = [-3,3,-6,0];
const sofa: Obstacle = {center:[0,.5,-3],size:[2,1,1],yaw:0};
describe('walkthrough containment', () => {
  it('prevents crossing furniture even with a large movement', () => {
    expect(moveWithinRoom([0,1.2,-1],[0,0,-4],limits,3,[sofa])[2]).toBeGreaterThan(-2.4);
  });
  it('keeps camera inside floor, ceiling and room', () => {
    expect(moveWithinRoom([0,1.2,-1],[10,-10,10],limits,3,[])).toEqual([3,.4,0]);
  });
  it('permits elevated inspection and escaping an invalid spawn', () => {
    expect(moveWithinRoom([0,2,-1],[0,0,-4],limits,3,[sofa])[2]).toBeCloseTo(-5);
    expect(moveWithinRoom([0,1.2,-3],[0,0,2],limits,3,[sofa])[2]).toBeCloseTo(-1);
  });
  it('excludes openings and malformed metadata', () => {
    expect(furnitureObstacles([{label:'door',center:[0,1,-3],size:[1,.1,2]}, {label:'sofa',center:[NaN,0,0],size:[1,1,1]}])).toEqual([]);
  });
});
