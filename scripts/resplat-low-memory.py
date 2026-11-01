"""Inference-only chunked cost-volume adapter; upstream checkout stays unchanged.

Computes the same channel sum/neighbor mean in FP32 in 16-channel chunks,
instead of allocating a full multi-gigabyte broadcast product. Releases the
consumed warp volume before depth upsampling and makes the existing mono-feature
resize work in either orientation. Input pixels and calibrated cameras stay intact.
"""
import inspect
import importlib.util
import json
import re
from pathlib import Path
import sys
import textwrap


def chunked_cost(ref, warped, channels, chunk=16):
    import torch
    scores=torch.zeros((warped.shape[0],warped.shape[1],*warped.shape[3:]),
                       dtype=warped.dtype,device=warped.device)
    for first in range(0,channels,chunk):
        last=min(channels,first+chunk)
        scores.add_((ref[:,None,first:last,None,:,:]*warped[:,:,first:last]).sum(2))
    return (scores/(channels**0.5)).mean(1)


def prepare_forward_source(original):
    """Patch only audited expressions in the pinned upstream forward method."""
    pattern=r'\(\s*\(ref_features\.unsqueeze\(-3\)\.unsqueeze\(1\) \* warped_tgt_features\)\.sum\(2\)\s*/ \(c\*\*0\.5\)\s*\)\.mean\(1\)'
    replaced,count=re.subn(pattern,'_one_beyond_chunked_cost(ref_features, warped_tgt_features, c)',original)
    if count!=1:
        raise ValueError('Pinned upstream cost-volume expression changed')
    replaced,count=re.subn(r'(?m)^([ \t]*)cost_volume = (_one_beyond_chunked_cost[^\n]+)$',
                          r'\1cost_volume = \2\n\1del warped_tgt_features',replaced)
    if count!=1:
        raise ValueError('Pinned upstream warp lifetime changed')
    orientation='''    assert ori_h <= ori_w
    if ori_w > self.max_mono_vit_input_size:
        resize_w = self.max_mono_vit_input_size // 14 * 14
        resize_h = int((ori_h / ori_w) * self.max_mono_vit_input_size) // 14 * 14
    else:
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14'''
    orientation_safe='''    longest = max(ori_h, ori_w)
    if longest > self.max_mono_vit_input_size:
        resize_h = int((ori_h / longest) * self.max_mono_vit_input_size) // 14 * 14
        resize_w = int((ori_w / longest) * self.max_mono_vit_input_size) // 14 * 14
    else:
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14'''
    if replaced.count(orientation)!=1:
        raise ValueError('Pinned upstream mono resize expression changed')
    return replaced.replace(orientation,orientation_safe,1)


def install(source):
    sys.path.insert(0,str(source))
    from src.model.encoder.unimatch.mv_unimatch import MultiViewUniMatch
    original=textwrap.dedent(inspect.getsource(MultiViewUniMatch.forward))
    replaced=prepare_forward_source(original)
    scope=dict(MultiViewUniMatch.forward.__globals__,_one_beyond_chunked_cost=chunked_cost)
    exec(compile(replaced,str(source/'src/model/encoder/unimatch/mv_unimatch.py')+' [chunked inference]','exec'),scope)
    MultiViewUniMatch.forward=scope['forward']
    print('Compatibility: FP32 cost-volume chunks, released warp volume, full-frame portrait support',flush=True)


def run_with_coverage(source):
    from resplat_context_selection import select_coverage_contexts
    import numpy as np
    path=source/'scripts/infer_colmap.py'
    spec=importlib.util.spec_from_file_location('one_beyond_resplat_inference',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    active={};reports=[];loaded_counts={}
    def capture(function):
        def wrapped(*args,**kwargs):
            result=function(*args,**kwargs);active['scene']=result;loaded_counts[function.__name__]=len(result['image_names']);return result
        return wrapped
    module.load_colmap_scene=capture(module.load_colmap_scene)
    module.subset_scene_data=capture(module.subset_scene_data)
    original=module.select_context_views
    def select(c2w,count,strategy='fps'):
        if strategy!='fps':return original(c2w,count,strategy)
        scene=active['scene']
        if not np.allclose(c2w,scene['c2w']):raise ValueError('Context poses differ from loaded calibration')
        if loaded_counts['load_colmap_scene']!=len(c2w):
            raise ValueError('Inference silently omitted registered camera views')
        indices,report=select_coverage_contexts(c2w,count,scene['point3D_ids'])
        report.update(loaded_registered_views=loaded_counts['load_colmap_scene'],candidate_view_count=len(c2w))
        report['selected_names']=[scene['image_names'][i] for i in indices]
        reports.append(report);return indices
    module.select_context_views=select
    module.main()
    if '--output_dir' in sys.argv and reports:
        output=Path(sys.argv[sys.argv.index('--output_dir')+1]);output.mkdir(parents=True,exist_ok=True)
        (output/'context-selection.json').write_text(json.dumps(reports[-1],indent=2))


if __name__=='__main__':
    source=Path(sys.argv[1]).resolve()
    sys.argv=[str(source/'scripts/infer_colmap.py')]+sys.argv[2:]
    install(source)
    run_with_coverage(source)
