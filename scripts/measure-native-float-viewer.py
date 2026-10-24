"""Matched native/browser pixels; appearance parity is not geometry accuracy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
out = root/'reports/quality-queue-oct24/native-float'
native = root/'reports/hall-stage-audit-oct24/current-native'
rows = []
for number in range(5):
    name = f'{number:05d}'
    reference = np.asarray(Image.open(native/f'inference/rendered/{name}.png').convert('RGB'), dtype=float)
    path = out/f'{name}.jpg.png'
    image = np.asarray(Image.open(path).convert('RGB'), dtype=float)
    assert image.shape == reference.shape
    mse = float(np.mean((image-reference)**2))
    row = dict(camera=name, mae=float(np.abs(image-reference).mean()), psnr_db=float(10*np.log10(255**2/mse)))
    baseline = native/f'browser/fixed-{name}.jpg.png'
    if baseline.exists():
        before = np.asarray(Image.open(baseline).convert('RGB'), dtype=float)
        row['baseline_psnr_db'] = float(10*np.log10(255**2/np.mean((before-reference)**2)))
        sheet = Image.new('RGB',(1920,736),'#202020')
        sheet.paste(Image.open(baseline).convert('RGB'),(0,32));sheet.paste(Image.open(path).convert('RGB'),(960,32))
        draw = ImageDraw.Draw(sheet);draw.text((12,10),'Before: Spark packed PLY',fill='white');draw.text((972,10),'After: full precision, same PLY and camera',fill='white')
        sheet.save(out/f'before-after-{name}.jpg',quality=95)
    rows.append(row)
pixels = lambda name: np.asarray(Image.open(out/name).convert('RGB'))
returned_delta=np.abs(pixels('00000.jpg.png').astype(float)-pixels('returned.png').astype(float))
wall=json.loads((out/'wall.json').read_text()) if (out/'wall.json').exists() else None
wall_control=None
if wall:
    probes={name:[Image.open(out/name).convert('RGB').getpixel((round(x),round(y))) for x,y,z in wall['locations']] for name in ['wall-off.png','wall-on.png']}
    off,on=probes['wall-off.png'],probes['wall-on.png']
    wall_control=dict(pixels=probes,behind_occluded=bool(off[0][0]>240 and max(on[0])-min(on[0])<5),front_preserved=bool(on[1][1]>240 and on[1][0]<10 and on[1][2]<10),synthetic_only=True)
    assert wall_control['behind_occluded'] and wall_control['front_preserved']
report = dict(matched_poses=rows, stationary_pixels_identical=bool(np.array_equal(pixels('settled.png'),pixels('idle.png'))),return_pixels_identical=bool(not np.any(returned_delta)),return_mae=float(returned_delta.mean()),return_max_channel_difference=float(returned_delta.max()),wall_control=wall_control,geometry_quality_accepted=False)
(out/'metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
