"""Matched-pose appearance comparison; not a geometry accuracy score."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1] / 'reports/hall-stage-audit-oct24/current-native'
results = []
for name in ['00000', '00002']:
    paths = [('Actual input', root / f'inference/input/{name}.png'),
             ('Native gsplat', root / f'inference/rendered/{name}.png'),
             ('Current Spark', root / f'browser/fixed-{name}.jpg.png'),
             ('Float PLY diagnostic', root / f'float-reference-{name}.jpg.png')]
    native = np.asarray(Image.open(paths[1][1]).convert('RGB'), dtype=np.float64)
    tiles = []
    for label, path in paths:
        image = Image.open(path).convert('RGB')
        pixels = np.asarray(image, dtype=np.float64)
        assert pixels.shape == native.shape, (path, pixels.shape, native.shape)
        mse = float(np.mean((pixels-native)**2))
        results.append(dict(camera=name, stage=label, size=image.size,
                            mae=float(np.abs(pixels-native).mean()),
                            psnr_db=10*np.log10(255**2/mse) if mse else None))
        tile = Image.new('RGB', (960, 736), '#202020')
        tile.paste(image, (0, 32))
        ImageDraw.Draw(tile).text((12, 10), label, fill='white')
        tiles.append(tile)
    sheet = Image.new('RGB', (1920,1472))
    for i,tile in enumerate(tiles): sheet.paste(tile, ((i%2)*960,(i//2)*736))
    sheet.save(root / f'matched-stage-comparison-{name}.jpg', quality=95)
(root/'parity-metrics.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
