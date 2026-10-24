"""Build a source-only local/cloud release. Never package photos or credentials."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import datetime

root=Path(__file__).resolve().parents[1]
release='photographic-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
files=list((root/'backend/reconstruction').glob('*.py'))
files += [root/'backend'/name for name in ['local_comparison_camera.py','local_reconstruct.py','local_splat_training.py','gaussian_job.py','serve_splat_review.py','review_local_reconstruction.py','test_photographic_fusion.py']]
files += [root/'backend'/name for name in ['test_joint_geometry.py','test_video_probe.py','test_geometry_diagnostics.py']]
files += [root/'backend'/name for name in ['test_video_lifecycle.py','test_capture_diagnostics.py']]
files += [root/'scripts'/name for name in ['splat-review.html','splat-solid.js','splat-navigation.js','splat-native-float.js','splat-surface-confidence.js','verify-shared-release.py']]
files += [root/'scripts'/name for name in ['validate-room-geometry.py','audit-sparse-geometry.py','check-video-preparation.py']]
files += [root/'scripts'/name for name in ['inspect-walkthrough.py','report-video-coverage.py']]
files += [root/'scripts'/name for name in ['build-vggt-preview.py','evaluate-vggt.py']]
files += [root/'backend'/name for name in ['test_vggt_local.py','test_vggt_preflight.py']]
files += [root/'backend'/name for name in ['app.py','job_runner.py','test_vggt_web.py']]
files += [root/name for name in ['CAPTURE_SOP.md','THIRD_PARTY_NOTICES.md']]
files += list((root/'frontend/dist').rglob('*'))
files = [p for p in files if p.is_file()]
manifest=dict(release=release,files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              training_tools='Existing Brush/COLMAP binaries are unchanged',notice='Shared finishing/viewer/video preparation and inactive experimental VGGT web integration for an existing backend installation; no photos, API keys, model weights or service activation. Runtime not proven by checksums.')
output=root/'reports/shared-release';output.mkdir(exist_ok=True)
archive=output/(release+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(root)).replace('\\','/'))
    payload=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('release.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
(output/'latest.json').write_text(json.dumps(dict(release=release,archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))
print(archive)
