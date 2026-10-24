"""Private, content-addressed reuse of completed photographic training.

Only exact input bytes/order, training implementation and tool fingerprints
match. Semantic alignment/cleanup always runs again for the current room.
Callers hold the photographic process lock while loading/publishing entries.
"""
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
TRAINING_FILES = ['backend/gaussian_job.py','backend/local_reconstruct.py',
    'backend/local_splat_training.py','backend/local_comparison_camera.py',
    'backend/serve_splat_review.py','backend/review_local_reconstruction.py',
    'backend/reconstruction/capture_intake.py']


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def training_key(files, tools, mode='companion'):
    payload=dict(schema=2, mode=mode, inputs=[dict(sha256=digest(path),suffix=Path(path).suffix.lower()) for path in files],
        implementation={name:digest(ROOT/name) for name in TRAINING_FILES}, tools={})
    for name,value in sorted(tools.items()):
        executable=Path(value)
        payload['tools'][name]=dict(path=str(executable.resolve()),sha256=digest(executable)) if executable.is_file() else str(value)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def verified_t4_key(files):
    """Separate provenance for a previously authorized, downloaded T4 result.

    Looking up this cache never starts or contacts a cloud machine. It cannot
    masquerade as an export produced by the local Windows executable.
    """
    return training_key(files,dict(processor='verified-t4-brush-0.3.0',
        recovery_worker=digest(ROOT/'scripts/t4-recovery-worker.py'),steps='2400',resolution='768',max_splats='50000'),
        mode='standalone-verified-t4')


def restore_verified_photo_set(cache_root, files, target, spaces_root):
    """Reuse exact photo multisets while preserving cached camera/image order.

    A shuffled upload is not a changed capture. Never sort the training dataset
    or reinterpret its camera IDs. Legacy ordered cache keys remain valid; the
    fallback derives them only from completed local photo jobs and re-hashes the
    original uploaded bytes. No cloud contact, pixel similarity, video reordering
    or trust in metadata's claimed input fingerprints.
    """
    files=list(map(Path,files));target=Path(target);spaces_root=Path(spaces_root)
    if not 3<=len(files)<=60:return False
    def identities(paths):return [(digest(path),path.suffix.lower()) for path in paths]
    expected=identities(files);direct=verified_t4_key(files)
    def load(key,original):
        if not restore(cache_root,key,target):return False
        proof=dict(processor='verified-t4',cache_key=key,input_order_changed=original!=expected,
                   requested_inputs=expected,trained_inputs=original,
                   notice='Exact photo multiset reuse; cached cameras and image names are unchanged. No fresh training or cloud use.')
        (target/'reuse-provenance.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
        return True
    if load(direct,expected):return True
    upload_root=(spaces_root.parent/'uploads').resolve()
    for metadata_path in sorted(spaces_root.glob('*/metadata.json')):
        try:
            metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata.get('status')!='completed' or metadata.get('input_type')!='photos':continue
            original=metadata.get('file_paths')
            if not isinstance(original,list) or len(original)!=len(files) or not all(isinstance(p,str) for p in original):continue
            original=list(map(Path,original))
            if not all(p.resolve().is_relative_to(upload_root) and p.is_file() for p in original):continue
            actual=identities(original)
            if Counter(actual)!=Counter(expected):continue
            key=verified_t4_key(original)
            if key!=direct and load(key,actual):return True
        except (OSError,ValueError,KeyError,TypeError):continue
    return False


def artifact_files(source):
    paths=[source/name for name in ['scene.ply','config.json','manifest.json','poses/inputs.json']]
    if (source/'poses/video_selection.json').is_file():paths.append(source/'poses/video_selection.json')
    for name in ['poses/images','poses/brush_dataset/images','poses/brush_dataset/sparse']:
        paths.extend(path for path in (source/name).rglob('*') if path.is_file())
    for model in source.glob('poses/model_text_*'):
        paths.extend(model/name for name in ['images.txt','cameras.txt','points3D.txt'])
    if not all(path.is_file() for path in paths) or not list(source.glob('poses/model_text_*/images.txt')):
        raise ValueError('Training cache requires completed camera and image evidence')
    for name in ['poses/images','poses/brush_dataset/images','poses/brush_dataset/sparse']:
        if not any(path.is_file() for path in (source/name).rglob('*')):
            raise ValueError('Incomplete training dataset')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='ready' or not 30<=manifest.get('vertices',0)<=1000000:
        raise ValueError('Incomplete training manifest')
    return paths


def publish(cache_root, key, source):
    source=Path(source);cache_root=Path(cache_root)
    paths=artifact_files(source)
    final=cache_root/key
    if final.exists():
        return False
    pending=cache_root/('.pending-'+uuid.uuid4().hex);pending.mkdir(parents=True)
    checksums={}
    for path in paths:
        name=path.relative_to(source).as_posix();destination=pending/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        checksums[name]=digest(destination)
    (pending/'cache.json').write_text(json.dumps(dict(schema=1,key=key,files=checksums)),encoding='utf-8')
    pending.rename(final)
    return True


def restore(cache_root, key, target):
    source=Path(cache_root)/key;target=Path(target)
    try:
        manifest=json.loads((source/'cache.json').read_text(encoding='utf-8'))
        if manifest.get('schema')!=1 or manifest.get('key')!=key:
            return False
        checksums=manifest['files']
        for name in ['scene.ply','config.json','manifest.json','poses/inputs.json']:
            if name not in checksums:return False
        for name,expected in checksums.items():
            path=source/name
            if not path.resolve().is_relative_to(source.resolve()) or not path.is_file() or digest(path)!=expected:
                return False
        artifact_files(source)
    except (OSError,ValueError,KeyError,TypeError):
        return False
    # Validate every byte before copying anything, so corrupt entries are misses.
    for name in checksums:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,destination)
    return True
