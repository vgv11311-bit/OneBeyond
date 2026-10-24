"""Keep this turn's superseded test exports on disk, but out of completed gallery."""
import json
from pathlib import Path
for job in ('refined-95d1e0ee33','refined-0d0ca04d3c','refined-1742f8446f','refined-fac6323b9f'):
    path=Path('storage/spaces')/job/'metadata.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    data['status']='superseded'
    data['stage']='Superseded development validation; original files retained'
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
