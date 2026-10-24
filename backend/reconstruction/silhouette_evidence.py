"""Measure whether an instance mask supports a detected outline.

Only geometry statistics are retained. Source pixels never become an asset.
"""

from pathlib import Path

import numpy as np


def assess_silhouettes(objects, mask_path, source_shape):
    path=Path(mask_path) if mask_path else None
    masks=None
    scores=None
    if path and path.exists():
        try:
            with np.load(path,allow_pickle=False) as data:
                masks=data['masks'].astype(bool)
                scores=data['scores'].astype(float) if 'scores' in data else None
        except (OSError,ValueError,KeyError):
            masks=None
    result=[]
    for index,obj in enumerate(objects):
        label=obj.label
        status='bbox_approximate'
        support=0.0
        score=None
        if masks is not None and index<len(masks):
            mask=masks[index]
            if scores is not None and index<len(scores) and np.isfinite(scores[index]):
                score=float(scores[index])
            height,width=mask.shape
            source_height=max(source_shape[0],1)
            source_width=max(source_shape[1],1)
            # Scene boxes are pixel coordinates in the source photo; the mask
            # worker may use a resized raster. The relative coordinates remain.
            x0=max(0,min(width,int(obj.bbox[0]/source_width*width)))
            y0=max(0,min(height,int(obj.bbox[1]/source_height*height)))
            x1=max(x0+1,min(width,int(obj.bbox[2]/source_width*width)))
            y1=max(y0+1,min(height,int(obj.bbox[3]/source_height*height)))
            # The mask is already box-prompted; its useful signal is whether a
            # nontrivial silhouette was returned, not a claim of semantic truth.
            area=max(1,(x1-x0)*(y1-y0))
            support=float(np.count_nonzero(mask[y0:y1,x0:x1]))/area
            if support>=.01 and np.count_nonzero(mask)>=12 and (score is None or score>=.35):
                status='sam2_silhouette'
        result.append({'index':index,'label':label,'outline':status,
                       'mask_coverage':round(support,4),
                       'mask_score':round(score,3) if score is not None else None})
    return result
