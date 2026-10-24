// Contain the source photograph: never crop its vertical field on wide screens.
export function photoVerticalFov(fy: number, photoAspect: number, viewportAspect: number) {
  return 2 * Math.atan(0.5 / fy * Math.max(1, photoAspect / viewportAspect)) * 180 / Math.PI;
}
