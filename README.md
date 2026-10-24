# One Beyond — source export

React/Vite frontend, Flask backend, reconstruction engine source and browser viewers.
This is a separate GitHub source copy. The working local installation is unchanged.

## Included

- `frontend/`: application source, public website assets, npm manifest and lockfile.
- `backend/`: Flask API, reconstruction source, tests and reusable library assets.
- `scripts/`: processing, rendering, evaluation and setup source utilities.
- `unity/`: optional viewer source/project settings, without player builds or room data.
- Third-party notices and an empty `.env.example`.

## Intentionally excluded

No existing Git history, credentials, actual `.env`, uploaded photos/videos, saved rooms,
private reports, datasets, backups, downloaded model weights, virtual environments,
`node_modules`, compiled frontend/player builds or caches are included.
The new copy starts with an empty saved-room collection. Do not copy the private data
back into this repository to make Recents look populated.

## Frontend build / GitHub import

Select **frontend** as the app root in a compatible Node.js hosting importer.

```sh
npm ci
npm run build
```

Build output: `frontend/dist` relative to repository root (`dist` from the app root).
Use `frontend/package-lock.json`; do not create a second unrelated root Node app.
Configure SPA route fallback to `index.html` for a static deployment.

**A frontend-only import does not deploy the Python API.** The frontend calls `/api`.
Production needs a compatible Python host serving both frontend and API, or explicit
same-origin routing/reverse proxy to the separately hosted API. A Node-only importer
cannot run the complete reconstruction backend just because this repo imports cleanly.
Domain/DNS migration and hosting purchases have not been performed by this export.

## Backend requirements

`requirements.txt` describes the web/API baseline, not the complete reconstruction
environment. Its clean-machine installation is not verified by the export process.
Local API entry point: `backend/app.py`. Configure secrets on the backend host, not in
frontend code. The production host needs a WSGI server and persistent private storage
for uploads/job metadata/results; a temporary build directory is insufficient.

Reconstruction workers additionally require separately configured COLMAP, Brush,
FFmpeg, Spark 2.3.1 assets and, for selected engines, CUDA/models or isolated Blender
and geometry environments. These binaries/weights are not in this repository.
Several worker launch paths remain specific to the existing Windows installation;
Linux/Hostinger compatibility must be tested rather than assumed. No server is
automatically started, and no cloud permission/lease is copied.

The renderer currently resolves Spark beneath the backend host's
`LOCALAPPDATA/OneBeyond/tools/spark-2.3.1/package/dist` (or home-directory equivalent).
Do not assume `npm ci` alone installs that external runtime.

## Privacy and deployment cautions

Create a **private GitHub repository first**. This export was screened for common
credential signatures and forbidden file categories; this is not a guarantee against
every possible secret, license issue or personal identifier in source comments/tests.
Historic source utilities may refer to missing local fixtures, paths or review data;
they are not a one-click deployment process and should not be run indiscriminately.

The source still needs a public-deployment security review (authentication, upload
privacy, per-user access, quotas and control of paid workers). Do not expose an
unrestricted reconstruction API to everyone merely to test a GitHub import.
The hall's reconstruction quality is still under review; source publication does not
mean the underlying distortion problem is solved.

See `THIRD_PARTY_NOTICES.md` and bundled asset attribution files. Optional model/tool
licensing must be checked before enabling or redistributing those components.
