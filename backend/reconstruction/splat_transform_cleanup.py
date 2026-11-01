"""Optional, bounded SplatTransform selection with exact-row preservation.

SplatTransform proposes which Gaussians to keep. Its rewritten attributes are
never used: retained rows come verbatim from our float32 PLY. Numerical guards
limit destructive edits; they do not establish visual or geometric accuracy.
"""
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .gaussian_geometry import read_ply,write_ply

VERSION='3.10.3'
INDEX='onebeyond_source_index'


def tiled_source(source):
    """Change dispatch partitioning only; voxel math and thresholds stay intact.

    The pinned upstream implementation batches up to 256 x 4096 blocks in a
    dispatch. Small integrated GPUs can exceed Windows' driver watchdog.
    Four blocks per axis and one batch keep the exact same voxel evaluation,
    with every Gaussian list retained. Unexpected upstream code is rejected.
    """
    replacements={'static MAX_BLOCKS_PER_BATCH = 4096;':'static MAX_BLOCKS_PER_BATCH = 64;',
                  'const batchSize = 16;':'const batchSize = 4;',
                  'const MEGA_MAX_BATCHES = 256;':'const MEGA_MAX_BATCHES = 1;',
                  'const MEGA_MAX_INDICES = 2 * 1024 * 1024;':'const MEGA_MAX_INDICES = 128 * 1024;'}
    for before,after in replacements.items():
        if source.count(before)!=1:raise ValueError('Pinned voxel dispatch implementation changed')
        source=source.replace(before,after)
    return source


def safe_cli(original_cli):
    bundled=Path(original_cli).parent.parent/'dist/cli.mjs'
    source=bundled.read_text(encoding='utf-8');modified=tiled_source(source)
    target=bundled.with_name('onebeyond-floater-tiles.mjs')
    if not target.is_file() or target.read_text(encoding='utf-8')!=modified:
        target.write_text(modified,encoding='utf-8')
    receipt=dict(version=VERSION,upstream_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 tiled_sha256=hashlib.sha256(modified.encode()).hexdigest(),
                 changes='Dispatch partition sizes only; no voxel resolution, opacity, contribution or Gaussian changes')
    (tool_home()/'floater-tiles-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    launcher=target.with_name('onebeyond-floater-cli.mjs')
    launcher.write_text("import { main } from './onebeyond-floater-tiles.mjs';\nawait main();\n",encoding='utf-8')
    return launcher


def tool_home():
    data=os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_DATA_HOME') or str(Path.home()/'.local/share')
    return Path(os.environ.get('ONE_BEYOND_SPLAT_TRANSFORM_HOME') or Path(data)/'OneBeyond/tools'/('splat-transform-'+VERSION))


def installation():
    home=tool_home();package=home/'node_modules/@playcanvas/splat-transform';node=shutil.which('node')
    try:
        manifest=json.loads((package/'package.json').read_text(encoding='utf-8'))
        if manifest.get('version')!=VERSION or manifest.get('license')!='MIT' or not node:
            return None
        cli=package/'bin/cli.mjs'
        return (node,cli) if cli.is_file() else None
    except (OSError,ValueError):return None


def tagged_input(header,values,destination):
    if INDEX in values.dtype.names:raise ValueError('Input already contains the provenance column')
    if len(values)>2**24:raise ValueError('Scene exceeds exact float32 provenance range')
    tagged=np.empty(len(values),dtype=[*values.dtype.descr,(INDEX,'<f4')])
    for name in values.dtype.names:tagged[name]=values[name]
    tagged[INDEX]=np.arange(len(values),dtype=np.float32)
    write_ply(destination,header.replace('end_header','property float '+INDEX+'\nend_header'),tagged)


def retained_indices(filtered,count):
    if INDEX not in filtered.dtype.names:raise ValueError('Filter discarded provenance; original retained')
    raw=filtered[INDEX]
    if not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():raise ValueError('Invalid source indices')
    ids=raw.astype(np.int64)
    if len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=count).any():raise ValueError('Duplicate or out-of-range source indices')
    return np.sort(ids)


def voxel_size(values):
    xyz=np.column_stack([values[k] for k in ('x','y','z')]).astype(float)
    scales=np.exp(np.clip(np.column_stack([values['scale_'+str(i)] for i in range(3)]),-30,20))
    # Include the full Gaussian footprint, not just its centre. Bound the GPU
    # grid independently of unknown SfM units; 0.05 is not necessarily 5 cm.
    radius=3*scales.max(axis=1)
    extent=(xyz+radius[:,None]).max(axis=0)-(xyz-radius[:,None]).min(axis=0)
    return max(float(extent.max())/192.,1e-5)


def assess(values,ids,cameras):
    count=len(values);removed=np.ones(count,bool);removed[ids]=False
    report=dict(input_splats=count,output_splats=len(ids),removed_splats=int(removed.sum()),
                removed_fraction=float(removed.mean()),maximum_removed_fraction=.03)
    if len(ids)<30 or report['removed_fraction']>.03:return False,dict(report,reason='Removal exceeds conservative budget')
    if not removed.any():return True,dict(report,reason='No floaters selected')
    alpha=1/(1+np.exp(-np.clip(values['opacity'].astype(float),-40,40)))
    # Strong opaque detail cannot silently disappear in a post-process.
    if (alpha[removed]>.15).any():return False,dict(report,reason='Filter selected opaque geometry; review required')
    xyz=np.column_stack([values[k] for k in ('x','y','z')]).astype(float)
    losses=[]
    for camera in cameras:
        p=(xyz-np.asarray(camera['position']))@np.asarray(camera['rotation'])
        w,h,fx,fy,cx,cy=camera['intrinsics'];z=-p[:,2]
        u=fx*p[:,0]/np.maximum(z,1e-9)+cx;v=-fy*p[:,1]/np.maximum(z,1e-9)+cy
        visible=(z>.01)&(u>=0)&(u<w)&(v>=0)&(v<h)
        losses.append(float(alpha[visible&removed].sum()/max(alpha[visible].sum(),1e-8)))
    maximum=max(losses,default=0.)
    report.update(maximum_camera_opacity_mass_loss=maximum,
                  camera_guard='Frustum-weighted opacity only; not an occlusion/visibility proof')
    if maximum>.01:return False,dict(report,reason='Camera opacity loss exceeds conservative budget')
    return True,dict(report,reason='Numerical guards passed; moving-view visual review still required')


def select_streak_candidates(values,tool_ids):
    """Require both low voxel contribution AND a translucent 1-D covariance.

    The voxel proposal alone can erase low-opacity surfaces in feed-forward
    scenes. A flat wall has two broad axes; comparing the TWO largest axes
    preserves it. This is a conservative heuristic, not semantic truth.
    """
    proposed=np.ones(len(values),bool);proposed[tool_ids]=False
    logs=np.sort(np.column_stack([values['scale_'+str(i)] for i in range(3)]),axis=1)
    alpha=1/(1+np.exp(-np.clip(values['opacity'].astype(float),-40,40)))
    needles=(logs[:,2]-logs[:,1]>np.log(20.))&(alpha<.15)
    selected=proposed&needles
    return np.flatnonzero(~selected),dict(tool_proposed_removals=int(proposed.sum()),
        protected_non_streak_splats=int((proposed&~needles).sum()),
        selection_policy='Voxel-rejected AND opacity < 0.15 AND largest/second axis > 20; planar splats protected')


def filter_export(path,config,*,enabled=True):
    """An optional cleanup failure must never discard a completed reconstruction."""
    try:
        return _filter_export(path,config,enabled=enabled)
    except Exception as error:
        return dict(tool='SplatTransform',version=VERSION,applied=False,status='unavailable',
            quality_accepted=False,requires_visual_review=True,geometry_repair_claimed=False,
            reason=type(error).__name__+': optional cleanup unavailable; original retained')


def _filter_export(path,config,*,enabled=True):
    path=Path(path).resolve();started=time.monotonic()
    report=dict(tool='SplatTransform',version=VERSION,applied=False,requires_visual_review=True,
                quality_accepted=False,geometry_repair_claimed=False,retained_attributes_modified=False)
    if not enabled:return dict(report,status='disabled',reason='Disabled for this reconstruction')
    tool=installation()
    if tool is None:return dict(report,status='unavailable',reason='Pinned MIT SplatTransform installation unavailable; original retained')
    header,values=read_ply(path)
    if not all(np.isfinite(values[name]).all() for name in values.dtype.names):
        return dict(report,status='rejected',reason='Input contains non-finite attributes; original retained')
    size=voxel_size(values);report.update(voxel_size_scene_units=size,voxel_grid_max_axis=192,
        opacity_cutoff=.1,minimum_contribution=1/255)
    inverse=np.linalg.inv(np.asarray(config['model_transform'],float))
    cameras=[dict(c,position=(inverse[:3,:3]@np.asarray(c['position'])+inverse[:3,3]).tolist(),
                  rotation=(inverse[:3,:3]@np.asarray(c['rotation'])).tolist()) for c in config['cameras']]
    work=path.parent/'floater-cleanup';work.mkdir(exist_ok=True)
    try:
        from colmap_prepare import run_stage
        with tempfile.TemporaryDirectory(prefix='candidate-',dir=work) as folder:
            temporary=Path(folder).resolve()
            if temporary.parent!=work.resolve():raise ValueError('Unexpected filter workspace')
            source=temporary/'tagged.ply';output=temporary/'filtered.ply';tagged_input(header,values,source)
            cli=safe_cli(tool[1])
            report['dispatch_partition']='4x4x4 blocks, one batch; voxel evaluation unchanged'
            with_guard=[tool[0],str(cli),str(source),'--filter-floaters',f'{size:.9g},0.1,{1/255:.9g}',str(output)]
            adapter=os.environ.get('ONE_BEYOND_SPLAT_TRANSFORM_GPU')
            if adapter:
                if not adapter.isdigit():raise ValueError('GPU adapter must be a nonnegative index')
                with_guard+=['--gpu',adapter]
            run_stage(work,'splat_transform_floaters',with_guard,90)
            _,filtered=read_ply(output);tool_ids=retained_indices(filtered,len(values))
            ids,selection=select_streak_candidates(values,tool_ids);report.update(selection)
            accepted,assessment=assess(values,ids,cameras);report.update(assessment)
            if not accepted:
                report['status']='rejected';return report
            if len(ids)==len(values):
                report['status']='unchanged';return report
            baseline=path.with_name(path.stem+'-before-floater-filter.ply')
            if baseline.exists():raise ValueError('Refusing to overwrite the preserved baseline')
            candidate=temporary/'exact-source-rows.ply';write_ply(candidate,header,values[ids])
            _,checked=read_ply(candidate)
            if checked.tobytes()!=values[ids].tobytes():raise ValueError('Retained attributes changed')
            shutil.copyfile(path,baseline);candidate.replace(path)
            report.update(applied=True,status='filtered',baseline_file=baseline.name)
            return report
    except Exception as error:
        report.update(status='rejected' if isinstance(error,ValueError) else 'unavailable',
                      reason=type(error).__name__+': filter did not finish safely; original retained')
        return report
    finally:
        report['seconds']=round(time.monotonic()-started,3)
        try:
            (work/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        except OSError:
            # A log-write failure must not misreport an already applied subset.
            report['report_write_failed']=True
