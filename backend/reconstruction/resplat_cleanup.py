"""Conservative geometry cleanup after covariance export, without sharpening."""
from pathlib import Path
import shutil
import numpy as np

from .gaussian_geometry import read_ply,write_ply,clean_splats


def clean_export(path,config):
    path=Path(path);header,values=read_ply(path)
    if not all(np.isfinite(values[name]).all() for name in values.dtype.names):
        raise ValueError('Nonfinite Gaussian export cannot be cleaned safely')
    inverse=np.linalg.inv(np.asarray(config['model_transform'],float))
    cameras=[]
    for camera in config['cameras']:
        cameras.append(dict(camera,position=(inverse[:3,:3]@np.asarray(camera['position'])+inverse[:3,3]).tolist(),
                            rotation=(inverse[:3,:3]@np.asarray(camera['rotation'])).tolist()))
    # Low opacity alone is not evidence of an artifact in feed-forward scenes.
    # Preserve that coverage; only the spatial/support tests may remove it.
    cleaned,report=clean_splats(values,cameras,minimum_alpha=0.)
    report.update(version=1,positions_modified=False,sharpening=False,blur_repair_claimed=False,
                  maximum_removed_fraction=.03)
    if len(cleaned)<len(values)*.97:
        report.update(applied=False,reason='Cleanup exceeded conservative removal budget; original retained')
        return values,report
    applied=report['removed_splats']>0 or report['needle_scales_capped']>0
    report['applied']=applied
    if applied:
        backup=path.with_name(path.stem+'-unfiltered.ply')
        if backup.exists():raise ValueError('Refusing to overwrite unfiltered baseline')
        shutil.copyfile(path,backup)
        temporary=path.with_name(path.stem+'-filtered.tmp')
        write_ply(temporary,header,cleaned)
        temporary.replace(path)
    return cleaned,report
