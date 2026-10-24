# Third-party notices

## Optional local reconstruction/review tools

The local experiment invokes official portable COLMAP 4.2.1 and Brush 0.3.0
binaries installed outside this repository. It reuses the existing FFmpeg
executable. These tools are not bundled into the production frontend.

Sources: https://github.com/colmap/colmap and
https://github.com/ArthurBrussee/brush and https://ffmpeg.org/ .
Retain the upstream licences when redistributing binaries.

The separate localhost QA viewer loads a locally installed, integrity-verified
Spark 2.3.1 module (MIT, https://github.com/sparkjsdev/spark) and the project's
existing Three.js installation. It does not replace the Unity/GLB viewer or
upload source photos. Spark's licence is retained in its installed package.
Optional HEIC decoding uses pillow-heif 1.8.1 and Pillow in `.venv-local-splats`,
not an upgrade of the production reconstruction environment.

## OpenMVS 2.4.0 - standalone local evaluation

Official source/release: https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0

License: GNU Affero General Public License v3.0; see upstream
https://github.com/cdcseacave/openMVS/blob/v2.4.0/LICENSE .

The unchanged portable CPU executables are installed outside this repository
under the user's local OneBeyond tools directory. Archive SHA-256:
`0c31660c15c9ebc4c106873cf67564d9570d404aef7a6403451da1b6178b2167`.
The installer and benchmark invoke separate CLI processes; no upstream source
is copied into the application and no primary pipeline switch or public cloud
deployment is implied. This notice is not a determination of the obligations
of a future combined/distributed/network service; that requires review and the
user's explicit adoption/distribution decision.

## VGGT: isolated joint-geometry evaluation preparation

Source: https://github.com/facebookresearch/vggt

Pinned revision: `a288dd0f14786c93483e45524328726ab7b1b4ce`.
Source-only checkout is outside this repository under the user's OneBeyond tools
directory; no upstream files or weights are bundled in the application release.
Upstream licensing: https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/LICENSE.txt

Approved isolated-evaluation checkpoint: https://huggingface.co/facebook/VGGT-1B-Commercial
at revision `ebb29a532abe92960eeb6903a5530f16990ef4ab`.
User access was approved. The pinned 5,026,367,224-byte checkpoint and its
LICENSE were downloaded locally on 22 October and verified against upstream
SHA256 `2b766b284359bc47ce26be107254621f685b758a0282082ff109f3ff02788b53`.
Local source/header/input preflight passed without model construction or
inference. Weights and credentials are excluded from application releases.
Use remains subject to its license and acceptable-use conditions; this notice
does not authorize commercial adoption or substitute research-only weights.

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
AnySplat remains source-only and inactive pending dependency/license clarification.
This notice is not a full commercial-license audit of all optional dependencies.

## 3D-RE-GEN: point-evidence quantile filter

Repository: https://github.com/cgtuebingen/3D-RE-GEN

Revision: `9a6275f3af97eae484c34d0ff5d9cc8177adaa17`

Upstream file: `src/scene_reconstruction/source/utils_SR/pc_utils.py`

Adapted function: `filter_points_by_quantile` in
`backend/reconstruction/point_evidence.py`.

Changes: NumPy CPU implementation instead of Torch; finite-row checks, input
validation, and a minimum-support fallback. No other models, weights, submodules,
or upstream pipeline are included. This notice is not a complete audit of existing
One Beyond dependencies.

MIT License

Copyright (c) 2026 Tobias Sautter / Computergraphics (University of Tübingen)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
