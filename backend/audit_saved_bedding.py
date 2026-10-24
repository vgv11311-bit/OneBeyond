"""Read-only regression diagnostics for arbitrary saved reconstruction jobs."""
import json
import sys
from pathlib import Path
import cv2
from reconstruction.wall_contacts import resolve_bed_headboard,enforce_contacts
from reconstruction.bedding_appearance import add_fabric_patterns

for arg in sys.argv[1:]:
    folder=Path(arg)
    scene=json.loads((folder/'blender_scene_input.json').read_text())
    image=cv2.imread(str(next((folder/'extracted_frames').glob('photo_*.jpg'))))
    specs=scene['objects']
    out,_=enforce_contacts(resolve_bed_headboard(specs),scene['room_dimensions'])
    patterned=add_fabric_patterns(out,image,folder/'sam2_masks.npz')
    print(json.dumps({'job':folder.name,'objects':[
        {k:o.get(k) for k in ('label','yaw_degrees','wall_contact','image_bbox','fabric_pattern_source')}
        for o in patterned if o['label'] in {'bed','pillow','blanket','duvet','throw'}]},indent=2))
