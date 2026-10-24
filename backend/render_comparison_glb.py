"""Render a saved GLB from an explicit comparison camera, without editing it.

Run using .venv-blender Python. Neutral comparison lighting differs from Unity;
this does not score the web renderer or claim calibrated room illumination.
"""
import argparse
import json
from pathlib import Path
import bpy
from mathutils import Matrix, Vector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--glb', type=Path, required=True)
parser.add_argument('--camera', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--width', type=int, default=512)
args = parser.parse_args()
camera = json.loads(args.camera.read_text())['camera']
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
# Ignore embedded lights so both dark upholstery and wall appearance remain
# inspectable under one documented, neutral light rig.
for obj in list(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
conversion = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
bpy.ops.object.camera_add()
obj = bpy.context.object
obj.location = conversion @ Vector(camera['position'])
obj.rotation_euler = (conversion @ Matrix(camera['rotation_matrix'])).to_euler()
height, width = camera['source_shape']
fx = camera['intrinsics'][0][0]
cx, cy = camera['intrinsics'][0][2], camera['intrinsics'][1][2]
obj.data.sensor_fit = 'HORIZONTAL'
obj.data.sensor_width = 36
obj.data.lens = 36*fx
obj.data.shift_x = .5-cx
obj.data.shift_y = (cy-.5)*height/width
scene = bpy.context.scene
scene.camera = obj
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 12
scene.cycles.use_denoising = True
scene.render.threads_mode = 'FIXED'
scene.render.threads = 4
scene.render.resolution_x = args.width
scene.render.resolution_y = round(args.width*height/width)
scene.render.resolution_percentage = 100
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.5, .5, .5, 1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value = .6
eye = obj.location.copy()
for offset, energy, size in [((0, 0, 1.0), 300, 3), ((-1, 1, .6), 150, 2)]:
    bpy.ops.object.light_add(type='AREA', location=eye+Vector(offset))
    light = bpy.context.object
    light.data.energy = energy
    light.data.shape = 'DISK'
    light.data.size = size
    light.rotation_euler = (obj.location+obj.rotation_euler.to_matrix()@Vector((0, 0, -3))-light.location).to_track_quat('-Z', 'Y').to_euler()
scene.view_settings.view_transform = 'Standard'
scene.render.filepath = str(args.output.resolve())
bpy.ops.render.render(write_still=True)
