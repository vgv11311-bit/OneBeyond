"""Rear-view void coverage only; not a geometry-accuracy benchmark."""
import json
from pathlib import Path
import cv2
import numpy as np

root=Path(__file__).resolve().parents[1]/'reports/gaussian-solid-oct19'
result={'notice':'Matched rear-view screenshot check, excluding UI header. Void coverage is not 3D accuracy.'}
for name in ['raw-rear','solid-rear']:
    image=cv2.imread(str(root/(name+'.png')))
    if image is None:raise ValueError('Missing browser capture '+name)
    rgb=image[200:-20,20:-20,::-1].astype(float)
    void=(np.max(np.abs(rgb-np.array([23,27,32])),axis=2)<4)|(rgb.max(axis=2)<5)
    result[name]={'void_fraction':round(float(void.mean()),6)}
(root/'rear-void-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if result['solid-rear']['void_fraction']>.01:raise SystemExit('Completed rear still contains significant void')
