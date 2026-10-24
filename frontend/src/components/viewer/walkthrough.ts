export type Vec3 = [number, number, number];
export type Obstacle = { center: Vec3; size: Vec3; yaw: number };

export function furnitureObstacles(objects: Array<Record<string, any>>): Obstacle[] {
  return objects.filter(o => /^(sofa|bed|cabinet|wardrobe|desk|table|coffee_table|nightstand|side_table|chair)$/.test(o.label)
    && Array.isArray(o.center) && o.center.length === 3 && o.center.every(Number.isFinite)
    && Array.isArray(o.size) && o.size.length === 3 && o.size.every((n: number) => Number.isFinite(n) && n > 0))
    .map(o => ({ center: o.center as Vec3, size: o.size as Vec3,
      yaw: Number.isFinite(o.yaw_degrees) ? o.yaw_degrees * Math.PI / 180 : 0 }));
}

/** Small substeps prevent tunnelling; axis sliding avoids sticking at corners.
 * Objects use the exporter convention: width, depth, height, with Y up.
 * Elevated inspection remains possible above furniture. */
export function moveWithinRoom(start: Vec3, delta: Vec3, limits: number[], height: number, obstacles: Obstacle[]): Vec3 {
  const p: Vec3 = [...start];
  const steps = Math.max(1, Math.ceil(Math.hypot(...delta) / .06));
  const blocked = (point: Vec3, o: Obstacle) => {
    if (point[1] - .65 >= o.center[1] + o.size[2] / 2 || point[1] + .12 <= o.center[1] - o.size[2] / 2) return false;
    const x = point[0] - o.center[0], z = point[2] - o.center[2];
    const localX = Math.cos(o.yaw) * x - Math.sin(o.yaw) * z;
    const localZ = Math.sin(o.yaw) * x + Math.cos(o.yaw) * z;
    return Math.abs(localX) < o.size[0] / 2 + .18 && Math.abs(localZ) < o.size[1] / 2 + .18;
  };
  for (let step = 0; step < steps; step++) {
    for (const axis of [0, 2, 1]) {
      const next: Vec3 = [...p];
      next[axis] += delta[axis] / steps;
      next[0] = Math.max(limits[0], Math.min(limits[1], next[0]));
      next[2] = Math.max(limits[2], Math.min(limits[3], next[2]));
      next[1] = Math.max(.4, Math.min(height - .2, next[1]));
      // Allow escape if the estimated initial camera is already inside furniture.
      if (!obstacles.some(o => !blocked(p, o) && blocked(next, o))) p[axis] = next[axis];
    }
  }
  return p;
}
