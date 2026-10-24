"""Inference-only chunked cost-volume adapter; upstream checkout stays unchanged.

Computes the same channel sum/neighbor mean in FP32 in 16-channel chunks,
instead of allocating a full multi-gigabyte broadcast product. No image resizing,
view dropping, model substitution, parameter edits or CPU geometry fallback.
"""
import inspect
from pathlib import Path
import runpy
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


def install(source):
    sys.path.insert(0,str(source))
    from src.model.encoder.unimatch.mv_unimatch import MultiViewUniMatch
    original=textwrap.dedent(inspect.getsource(MultiViewUniMatch.forward))
    expression='''(
                (ref_features.unsqueeze(-3).unsqueeze(1) * warped_tgt_features).sum(2)
                / (c**0.5)
            ).mean(1)'''
    # inspect.dedent removes the class method's four leading spaces.
    expression=textwrap.dedent(expression)
    import re
    pattern=r'\(\s*\(ref_features\.unsqueeze\(-3\)\.unsqueeze\(1\) \* warped_tgt_features\)\.sum\(2\)\s*/ \(c\*\*0\.5\)\s*\)\.mean\(1\)'
    replaced,count=re.subn(pattern,'_one_beyond_chunked_cost(ref_features, warped_tgt_features, c)',original)
    if count!=1:
        raise ValueError('Pinned upstream cost-volume expression changed')
    scope=dict(MultiViewUniMatch.forward.__globals__,_one_beyond_chunked_cost=chunked_cost)
    exec(compile(replaced,str(source/'src/model/encoder/unimatch/mv_unimatch.py')+' [chunked inference]','exec'),scope)
    MultiViewUniMatch.forward=scope['forward']
    print('Compatibility: identical FP32 cost-volume formula, 16-channel temporary chunks',flush=True)


if __name__=='__main__':
    source=Path(sys.argv[1]).resolve()
    sys.argv=[str(source/'scripts/infer_colmap.py')]+sys.argv[2:]
    install(source)
    runpy.run_path(sys.argv[0],run_name='__main__')
