# One Beyond — COLMAP + ReSplat

A room reconstruction application with a React/TypeScript viewer and a Flask API.
COLMAP + ReSplat is the single reconstruction engine. The API preserves uploads,
reports processing progress and serves Gaussian PLY results for interactive review.

## Project structure

```text
backend/app.py                 HTTP API, upload/job lifecycle and asset delivery
backend/colmap_prepare.py      feature matching, calibration and undistortion
backend/reconstruction/       ReSplat dispatch, validation and navigation support
frontend/src/                 application and viewer interface
scripts/                      inference adapters, viewer renderer and setup utility
docs/                         configuration and dependency notes
```

## Requirements

- Python 3.10 or 3.11, Node.js 20+, npm, Git and FFmpeg/ffprobe.
- COLMAP available on PATH or through `ONE_BEYOND_COLMAP`.
- A compatible NVIDIA CUDA runtime for ReSplat inference. The web API and viewer
  can start without a GPU, but reconstruction cannot.
- Install the pinned ReSplat environment and checkpoint separately. Model weights,
  CUDA libraries, uploaded footage and generated scenes are not bundled in the source archive.

## Start locally

From the repository root, create and activate an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python backend/app.py
```

Open `http://127.0.0.1:5000`. On Linux, activate with `source .venv/bin/activate`.
The production frontend is served by Flask after the build; `npm run dev` is
available for frontend development. Keep runtime credentials outside the repository.

## Reconstruction flow

1. Upload overlapping room photographs or a fixed-lens walkthrough video.
2. Validate inputs and select frames; solve and undistort cameras with COLMAP.
3. Run ReSplat on the configured CUDA worker and validate the exported Gaussians.
4. Open the photographic viewer, inspect moving views, then save the reviewed result.

Keep the server storage directory durable: it holds upload data, job state,
camera outputs and generated assets. A source checkout alone does not restore
previously generated rooms. Never expose the development API directly to the
internet without authentication, upload quotas and appropriate network protection.

## Validation and limitations

Frontend build: `cd frontend; npm run build`. Python checks: see `VALIDATION.md`.
Local import/unit/build checks do not certify a complete GPU reconstruction.
Textureless surfaces, reflections, blur and missing viewpoint overlap can prevent
camera alignment or leave incomplete geometry. Scene scale and collision boundaries
are estimates. Review output before presenting it as a finished room.

## Configure ReSplat

`python scripts/prepare-resplat.py` downloads the pinned upstream source and
8-view checkpoint and records its integrity receipt. This step requires internet
access and substantial disk space; it does not install CUDA or start inference.
Create a separate CUDA environment using the requirements of the pinned source
and set `ONE_BEYOND_RESPLAT_PYTHON` to its Python executable.

| Variable | Purpose |
| --- | --- |
| `ONE_BEYOND_RESPLAT_SOURCE` | Pinned upstream checkout |
| `ONE_BEYOND_RESPLAT_WEIGHTS` | Checkpoint beside its generated `receipt.json` |
| `ONE_BEYOND_RESPLAT_PYTHON` | Python executable in the CUDA environment |
| `ONE_BEYOND_RESPLAT_ENABLED=1` | Enable the configured local CUDA worker |
| `ONE_BEYOND_COLMAP` | COLMAP executable override |
| `ONE_BEYOND_FFMPEG` | FFmpeg executable override |
| `ONE_BEYOND_SPARK_DIST` | Optional renderer distribution override |

The default source pin is `cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1`.
This package uses the matching `dl3dv_8v_512x960` inference preset. It accepts
5–10 photos or one 1–90 second video. Runtime limits are not a guarantee of
successful reconstruction within a fixed time.

Remote execution is optional and requires an already-running, explicitly
configured worker and verified connection. See `docs/RUNTIME.md`.
See `THIRD_PARTY_NOTICES.md` for external component attribution.
