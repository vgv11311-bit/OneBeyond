"""Headless geometry review only; lighting differs from the browser."""
import sys,math,json
from pathlib import Path
import bpy
from mathutils import Vector

folder=Path(sys.argv[1]).resolve()
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(folder/'model.glb'))
meta=json.loads((folder/'metadata.json').read_text())
camera_info=meta['metrics']['extra_details']['lighting']['projection_camera']
bpy.ops.object.camera_add(location=(0,0,camera_info.get('height',1.2)))
camera=bpy.context.object
camera.rotation_euler=Vector((0,1,0)).to_track_quat('-Z','Y').to_euler()
camera.data.lens=36*camera_info['intrinsics'][0][0]; camera.data.sensor_width=36
scene=bpy.context.scene; scene.camera=camera
scene.render.engine='CYCLES'; scene.cycles.samples=12
scene.render.resolution_x=810; scene.render.resolution_y=540; scene.render.resolution_percentage=100
scene.world.color=(.4,.4,.4)
for pos,power in [((0,1,2.5),500),((-2,2,2.2),350)]:
    bpy.ops.object.light_add(type='AREA',location=pos)
    lamp=bpy.context.object; lamp.data.energy=power; lamp.data.shape='DISK'; lamp.data.size=3
scene.render.filepath=str(folder/'geometry-review.png')
bpy.ops.render.render(write_still=True)
