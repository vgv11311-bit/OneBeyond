"""Export an existing room for private offline replay, never a new reconstruction.

No raw photographs, credentials, checkpoints or cloud APIs are included. The
textured mesh is nevertheless private visual room data; do not publish it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def self_contained_glb(path):
    with path.open('rb') as stream:
        magic, version, length = struct.unpack('<4sII', stream.read(12))
        size, kind = struct.unpack('<II', stream.read(8))
        if magic != b'glTF' or version != 2 or length != path.stat().st_size or kind != 0x4E4F534A:
            raise ValueError('Invalid GLB')
        if size > 10 * 1024**2:
            raise ValueError('GLB JSON exceeds budget')
        document = json.loads(stream.read(size))
    for item in document.get('buffers', []) + document.get('images', []):
        if 'uri' in item and not item['uri'].startswith('data:'):
            raise ValueError('External GLB dependency is not portable')


def assets(config):
    layers = ([config['fused_surface']] if config.get('fused_surface') else
              config.get('surfaces') or ([config['surface']] if config.get('surface') else []))
    result = [item['asset'] for item in layers]
    if not layers:
        result.append('scene.ply')
    if config.get('solid', {}).get('enabled'):
        result.append(config['solid']['glb_asset'])
    for name in result:
        path = PurePosixPath(name)
        if path.name != name or '\\' in name or ':' in name or path.suffix not in {'.glb', '.ply'}:
            raise ValueError('Unsupported room asset')
    return sorted(set(result))


def build(revision, output, three, spark):
    revision = revision.resolve()
    config_bytes = (revision / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    selected = assets(config)
    for name in selected:
        source = revision / name
        if not source.is_file() or source.is_symlink():
            raise ValueError('Missing or linked room asset')
        if source.suffix == '.glb':
            self_contained_glb(source)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(revision / 'config.json', output / 'config.json')
    for name in selected:
        shutil.copy2(revision / name, output / name)
    for source, target in [('splat-review.html', 'index.html'), ('splat-solid.js', 'support.js'),
                           ('splat-navigation.js', 'navigation.js'),
                           ('splat-surface-confidence.js', 'surface-confidence.js'),
                           ('serve-portable-room.py', 'serve.py')]:
        shutil.copy2(ROOT / 'scripts' / source, output / target)
    # Preserve installed dependency versions and notices; no CDN requests.
    for part in ['build', 'src', 'examples/jsm']:
        shutil.copytree(three / part, output / 'vendor/three' / part,
                        ignore=shutil.ignore_patterns('*.map'))
    shutil.copy2(three / 'LICENSE', output / 'vendor/three/LICENSE')
    (output / 'vendor/spark').mkdir(parents=True)
    shutil.copy2(spark / 'dist/spark.module.js', output / 'vendor/spark/spark.module.js')
    shutil.copy2(spark / 'LICENSE', output / 'vendor/spark/LICENSE')
    if (spark / 'dist/assets').is_dir():
        shutil.copytree(spark / 'dist/assets', output / 'vendor/spark/assets')
    (output / 'README.txt').write_text(
        'PRIVATE SAVED-ROOM REPLAY - NOT A NEW OR QUALITY-ACCEPTED RECONSTRUCTION\n'
        'This includes private room imagery embedded in the mesh. Do not publish.\n'
        'Requires Python 3.10+ and a WebGL2 browser, but no CUDA, cloud, API or internet.\n'
        'Run: python serve.py\nThen open http://127.0.0.1:8768\n'
        'Verification only: python serve.py --verify-only\n'
        'Existing distortion remains. Unobserved enclosure is inferred, not recovered.\n', encoding='utf-8')
    files = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = dict(schema=1, files=files, config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    fresh_reconstruction=False, quality_accepted=False, private_room_data=True,
                    raw_photos_included=False, cloud_required=False,
                    source_revision=revision.name)
    (output / 'bundle.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError('Will not replace existing bundle archive')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                target.write(path, output.name + '/' + path.relative_to(output).as_posix())
    return dict(status='private_replay_exported', files=len(files), archive=str(archive),
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                bytes=archive.stat().st_size, quality_accepted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.revision, args.output, ROOT / 'frontend/node_modules/three',
                           Path(os.environ['LOCALAPPDATA']) / 'OneBeyond/tools/spark-2.3.1/package'), indent=2))
