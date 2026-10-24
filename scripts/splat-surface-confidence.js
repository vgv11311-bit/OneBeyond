// A depth surface estimated from one photo is reliable near that capture,
// not from every camera. Fade it smoothly while retaining the trained splats.
export function surfaceWeight(position, forward, anchor, cameras) {
  if (!anchor || !Array.isArray(cameras) || cameras.length < 2) return 0;
  const distance = p => Math.hypot(...position.map((v, i) => v - p[i]));
  const own = distance(anchor.position);
  const other = Math.min(...cameras.filter(c => c.name !== anchor.name).map(c => distance(c.position)));
  const smooth = (low, high, value) => {
    const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
    return t * t * (3 - 2 * t);
  };
  const dominance = own < 1e-9 ? 1 : other / (own + other);
  const direction = forward.reduce((sum, v, i) => sum - v * anchor.rotation[i][2], 0);
  const weight = smooth(.5, .85, dominance) * smooth(.5, .95, direction);
  return Number.isFinite(weight) ? weight : 0;
}
