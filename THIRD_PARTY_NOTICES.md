# Third-party notices

COLMAP (https://github.com/colmap/colmap), FFmpeg (https://ffmpeg.org/),
ReSplat and its CUDA dependencies are installed separately, not bundled.
Retain upstream licenses when redistributing their source or binaries.

The viewer uses Spark 2.3.1 (MIT, https://github.com/sparkjsdev/spark),
Three.js and React. npm packages retain their own license files.
Optional HEIC decoding uses Pillow and pillow-heif.

## ReSplat: isolated native-Gaussian experiment

Source: https://github.com/cvg/resplat at
`cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1`.
MIT License, Copyright (c) 2026 Haofei Xu. The full upstream LICENSE is retained
with both source installations; source and checkpoint are not bundled in releases.
Checkpoint: `haofeixu/resplat`, revision
`c69ad72a49b8358ac46599cf74468090425ed9c7`, model-card license label MIT.
`resplat-base-dl3dv-512x960-view8-8179ed87.pth`, SHA256
`8179ed879d9f1d88abd2f3374af61ca6ccdfe6121c822f87cdf55ba8cbcfff59`.
The inference-only low-memory adapter changes the channel-reduction evaluation
order, not learned parameters, source photos, view count or image resolution.
The original pinned source remains unchanged. gsplat v1.5.3 is installed in the
isolated processing environment under its Apache-2.0 license.
This notice is not a full commercial-license audit of all optional dependencies.

