# Viewer visual regression contract

Portion 1 preserves the existing room reconstruction and defines a stable
renderer baseline before the later WebGPU renderer work begins.

## Golden views

Capture these views for both the sample room and one completed uploaded room:

1. Entrance: default camera at `[0, 1.2, 0]`, cinematic lighting.
2. Room center: walk forward for three seconds, then face the principal bed or desk.
3. Opposite wall: face the entrance so the rear wall and door are visible.
4. Lighting: repeat the entrance view in studio and daylight modes.

Use Balanced quality as the cross-device golden. Save the diagnostics baseline
after the model has loaded and FPS has stabilized for at least five seconds.

## Required invariants

- Room bounds, object transforms and camera start do not change between graphics profiles.
- WASD, sprint, mouse look, elevation and reset remain functional.
- Compatibility mode disables real-time shadows but does not hide geometry.
- WebGPU is labelled `Available` only as a capability; the active backend remains
  WebGL2/WebGL1 until the WebGPU renderer is deliberately enabled in a later portion.
- No frame may be accepted as a golden while the loading overlay is visible.

## Performance gates

- Balanced profile target: at least 50 FPS and p95 frame time at most 25 ms on the development laptop.
- Compatibility profile target: at least 50 FPS and p95 frame time at most 25 ms on integrated graphics.
- A regression is flagged if model-load time, draw calls, triangle count, texture count,
  heap use, or p95 frame time rises by more than 20% without an intentional scene change.

The live baseline button stores a versioned record per job and resolved quality
in browser local storage. The committed JSON in `baselines/` is the comparison
reference for the furnished sample room and should change only after an
intentional, reviewed renderer or scene update.
