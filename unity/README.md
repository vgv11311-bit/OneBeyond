# One Beyond Unity web viewer

This is the source project for a Unity 6000.6.6f1 / URP 17.6 viewer. It uses
Unity glTFast 6.20 to load existing same-origin room GLBs at runtime. Unity is
compiled once per viewer release, never per uploaded photograph. The Python /
Blender reconstruction backend is unchanged. Use the main checkout, not the
October 17 backup or the separate UI worktree.

## Build and publish

Activate an eligible license personally in Unity Hub; no credentials belong in
this repository. Install the Editor and Web Build Support, then run from the
main checkout:

```powershell
.\scripts\build-unity.ps1
```

The Editor generates its scene, pipeline and volume assets from source, builds
into `frontend/public/unity`, and writes `viewer.json` only on build success.
The script builds the frontend so Flask's existing static host serves it.
The generated player uses gzip with Unity's decompression fallback for static
hosts that do not set Content-Encoding. Normal HTTPS hosting remains required
for public distribution. No separate graphics server is needed.

## Interface contract (UI chat)

The App uses `RoomViewer`; it keeps Three.js when no Unity release is published.
`?renderer=three` explicitly selects the original viewer. A published Unity
viewer offers a manual Three.js fallback on startup/model errors. Existing
point-cloud and wireframe functions remain in Three.js; they are disabled,
not falsely advertised, in Unity. Shader/material fidelity must be checked in
a real browser before claiming visual improvement.

`unityBridge.ts` owns a versioned manifest and the runtime messages. React calls
`SendMessage('OneBeyondViewer', method, string)` for `LoadScene`, `SetQuality`,
`SetLighting`, `ResetCamera`, `SetFocused`, `SetMotion`, `Look`, and `Capture`.
The `.jslib` dispatches `one-beyond-unity` events on the originating canvas:
`ready`, `loaded`, `error`, `quality`, `metrics`, `screenshot`. Loaded/model
events carry a unique room session; stale sessions are ignored. Player lifetimes
are serialized and Quit is awaited to avoid multiple WebAssembly heaps.

Quality identifiers remain `auto`, `high`, `balanced` (medium), `compatibility`
(low) for compatibility with existing settings. Auto starts at medium and uses
five-second frame-rate windows: immediate downgrade below 27 FPS, upgrade only
after three windows above 55 FPS. It never changes room geometry or camera pose.
Imported glTF mirrors X in Unity. The camera, normals/orientation and collision
obstacles are converted consistently; model transforms are not rescaled or
repositioned. Source-photo FOV and aspect are preserved.

## Acceptance gate

1. Unity compilation and Web build succeed; no missing shaders/textures.
2. Browser loads actual saved hall and bedroom GLBs; compare the same camera.
3. Check materials, room orientation, near-plane clipping, fixture lighting.
4. Check WASD, drag/pointer-lock, Escape, reset, mobile controls and screenshot.
5. Change rooms repeatedly, exit/re-enter, and check memory is released.
6. Check Auto and manual modes, WebGL failures, model failures and fallback.
7. Record startup size/time and FPS; do not interpret unit tests as visual proof.

The October 17 missing-Package-Manager installation failure was repaired on
October 18. The Unity Web build and frontend build now succeed. Chrome checks
load and render saved hall and bedroom models, exercise quality/lighting/reset,
export screenshots, and exit. Both rooms additionally pass keyboard movement,
pixel-identical reset and exit/reopen checks. See the integration report for
evidence and remaining checks. These checks do not establish exact photo
reconstruction, mobile support or a universal quality/performance improvement.

The glTFast 6.20 importer still warns that `KHR_materials_emissive_strength` is
unsupported internally. `GlbEmission` now reads that optional metadata from the
same downloaded GLB and applies it to the imported material; bedroom browser
events confirm two emission-strength repairs. Runtime-loaded rooms also receive
an explicit neutral diffuse ambient probe. The original Three.js viewer remains
available. Chrome emits some framebuffer warnings on
Unity Quit; reopening the hall was successful despite those warnings.

Run `scripts/check-unity-browser.cjs` with Playwright available to exercise a
completed saved room. `ONE_BEYOND_CHECK_JOB` selects a saved job for validation,
and `ONE_BEYOND_CHECK_LABEL` names its report files. These are test inputs,
never source-photo-specific reconstruction rules.
