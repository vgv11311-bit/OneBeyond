"""Refresh cloth in a specifically supplied local refinement, preserving other objects."""
import copy,json,sys
from pathlib import Path
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.cloth_fit import fit_cloth
from reconstruction.blender_exporter import BlenderSceneExporter

folder=Path(sys.argv[1]); p=json.loads((folder/'blender_scene_input.json').read_text())
m=json.loads((folder/'metadata.json').read_text()); lighting=m['metrics']['extra_details']['lighting']
for obj in p['objects']:
    # Migrate only accessories emitted by the first diagnostic exporter.
    if 'color_rgb' in obj:
        obj['color']=obj.pop('color_rgb'); obj['color_source']='uploaded_photo_region'
        ih,iw=lighting['projection_camera']['source_shape']
        old_height=obj['size'][2]; obj['size'][2]*=ih/iw
        obj['center'][1]+=(obj['size'][2]-old_height)/2
prepared,_=prepare_scene_specs(copy.deepcopy(p['objects']),p['room_dimensions'])
objects=[new if old['label'] in {'throw','blanket','duvet'} else old for old,new in zip(p['objects'],prepared)]
objects=fit_cloth(objects,lighting.get('projection_camera'))
BlenderSceneExporter.create_room_glb(str(folder/'model.glb'),p['room_dimensions'],objects,lighting)
m['metrics']['extra_details']['scene_objects']=objects
(folder/'metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
print('Updated cloth:',folder.name,flush=True)
