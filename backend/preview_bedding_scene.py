"""Zero-provider-call bedding-only review; all other saved objects stay unchanged."""
import argparse
import copy
import json
import tempfile
import cv2
from pathlib import Path
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.cloth_fit import fit_cloth
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.lamp_profile import add_lamp_profiles
from reconstruction.cloth_extent import measure_cloth_extent

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);args=p.parse_args()
    payload=json.loads((args.source/'blender_scene_input.json').read_text())
    meta=json.loads((args.source/'metadata.json').read_text())
    specs=copy.deepcopy(payload['objects'])
    lighting=meta['metrics']['extra_details']['lighting']
    selected=[i for i,o in enumerate(specs) if o['label'] in {'bed','duvet','blanket','throw'}]
    repaired,_=prepare_scene_specs([specs[i] for i in selected],payload['room_dimensions'],camera=lighting.get('projection_camera'))
    # Keep accepted bed geometry and all unrelated objects. Only cloth changes.
    for j,i in enumerate(selected):
        if specs[i]['label']=='bed': repaired[j]=copy.deepcopy(specs[i])
    for j,i in enumerate(selected):
        if specs[i]['label']!='bed': specs[i]=repaired[j]
    specs=measure_cloth_extent(specs,args.source/'moge_geometry.npz',args.source/'sam2_masks.npz',lighting.get('projection_camera'))
    specs=fit_cloth(specs,lighting.get('projection_camera'))
    image=cv2.imread(meta['file_paths'][0])
    specs=add_lamp_profiles(specs,image,args.source/'sam2_masks.npz',{'floor_lamp'},source_shape=lighting.get('projection_camera',{}).get('source_shape'))
    output=Path(tempfile.mkdtemp(prefix='one-beyond-bedding-review-'))
    payload['objects']=specs
    (output/'blender_scene_input.json').write_text(json.dumps(payload,indent=2))
    BlenderSceneExporter.create_room_glb(str(output/'model.glb'),payload['room_dimensions'],specs,lighting,preview_path=str(output/'preview.png'))
    print(json.dumps({'output':str(output),'preview':str(output/'preview.png'),'provider_calls':0,'listed_in_recent':False}),flush=True)
