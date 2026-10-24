"""Measure real exact-input cache reuse; never claims a fresh training run."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from local_reconstruct import tool
from reconstruction.gaussian_cache import training_key,publish,digest
from reconstruction.gaussian_pipeline import build_companion
from reconstruction.base_engine import ReconstructionMetrics,ReconstructionResult

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--room',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-live-cache',action='store_true',help='Also seed private app cache from this completed local training artifact')
    args=parser.parse_args();source=args.source.resolve();room=args.room.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    for name in ['model.glb','metadata.json','moge_geometry.npz']:shutil.copyfile(room/name,output/name)
    saved=json.loads((room/'metadata.json').read_text(encoding='utf-8'))
    fields=ReconstructionMetrics.__dataclass_fields__
    metrics=ReconstructionMetrics(**{key:value for key,value in saved['metrics'].items() if key in fields})
    photos=sorted((source/'inputs').iterdir());key=training_key(photos,{name:tool(name) for name in ['brush','colmap']})
    start=time.monotonic();seeded=publish(output.parent/'gaussian-cache',key,source);seed_seconds=time.monotonic()-start
    live_seeded=publish(ROOT/'backend/storage/spaces/gaussian-cache',key,source) if args.seed_live_cache else False
    start=time.monotonic()
    result=build_companion('private-cache-benchmark',photos,output,lambda *args:print(args,flush=True),
        ReconstructionResult(True,'private-cache-benchmark',str(output/'model.glb'),metrics,preview_image_path=str(saved['file_paths'][0])))
    seconds=time.monotonic()-start
    if not result.get('cache_hit') or result.get('status')!='ready':raise RuntimeError('Cache reuse failed')
    raw=next((output/'gaussian').glob('*/raw-scene.ply'))
    identical=digest(raw)==digest(source/'scene.ply')
    if not identical:raise RuntimeError('Cache changed raw trained data')
    report=dict(status='passed',seeded=seeded,live_seeded=live_seeded,seed_seconds=round(seed_seconds,3),warm_companion_seconds=round(seconds,3),
        original_companion_seconds=701.844,raw_byte_identical=identical,descriptor=result,
        notice='Existing completed training seeded explicitly, then real cache restore and current geometry cleanup. Not a cold run or end-to-end Groq benchmark.')
    (ROOT/'reports/photographic-polish-oct19/cache-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
