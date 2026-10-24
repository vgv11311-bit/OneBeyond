import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/hall-resplat-audit-oct24'
audit=json.loads((out/'audit.json').read_text());baseline=out/'baseline'
hashfile=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
original=ROOT/'backend/storage/spaces/bf37e7eb-f8f'
checks=dict(metadata_unchanged=hashfile(original/'metadata.json')==audit['baseline_hashes']['metadata.json'],
            config_unchanged=hashfile(original/'gaussian'/audit['baseline_revision']/'config.json')==audit['baseline_hashes']['config.json'],
            ply_unchanged=hashfile(original/'gaussian'/audit['baseline_revision']/'scene.ply')==audit['baseline_hashes']['scene.ply'],
            photos_unchanged=all(hashfile(Path(p['path']))==p['sha256'] for p in audit['inputs']))
records=json.loads((out/'renderer-probes.json').read_text());metrics={}
# Exclude UI: this is pixel repeatability, not a perceptual-quality score.
read=lambda p:np.asarray(Image.open(p).convert('RGB'),float)[350:800,350:1200]
first=read(out/'baseline-entry.png')
for record in records:
 label=record['variant'];a=read(out/f'{label}-entry.png');b=read(out/f'{label}-return.png')
 metrics[label]=dict(entry_difference_from_baseline=float(np.abs(a-first).mean()),entry_return_mae=float(np.abs(a-b).mean()),buffer=record['poses'][0]['renderer']['buffer'],errors=record['errors'])
report=dict(preservation=checks,renderer_comparisons=metrics,quality_success=False,cpu_reference_excluded='Experimental CPU raster diagnostic did not reproduce the validated Spark image; its outputs are NOT quality evidence or a confirmed diagnosis.')
(out/'validation.json').write_text(json.dumps(report,indent=2))
sheet=Image.new('RGB',(1280,1080),'#111');draw=ImageDraw.Draw(sheet)
for row,pose in enumerate(['entry','right-rotation','rear']):
 for col,label in enumerate(['before','after']):
  image=Image.open(out/f'{label}-{pose}.png').convert('RGB');image=ImageOps.contain(image,(640,330))
  x=col*640;y=row*360;sheet.paste(image,(x,y+25));draw.text((x+8,y+5),f'{label}: {pose} - identical camera; quality NOT accepted',fill='white')
sheet.save(out/'before-after-comparison.jpg')
print(json.dumps(report,indent=2))
