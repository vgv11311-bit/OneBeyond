"""Evaluate saved masks and geometry without changing a saved room."""
import json,sys,tempfile,shutil
from pathlib import Path
from reconstruction.self_correction import correct_scene,Silhouettes
from reconstruction.blender_exporter import BlenderSceneExporter

for name in sys.argv[1:]:
    source=Path(name)
    data=json.loads((source/'metadata.json').read_text())['metrics']['extra_details']
    payload=json.loads((source/'blender_scene_input.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.glb'; shutil.copyfile(source/'model.glb',path)
        camera=data['lighting']['projection_camera']
        out,report=correct_scene(path,payload['objects'],payload['room_dimensions'],camera,source/'sam2_masks.npz',budget_seconds=8,max_trials=60)
        if report.get('accepted'):
            rebuilt=Path(tmp)/'rebuilt'/'model.glb'
            BlenderSceneExporter.create_room_glb(str(rebuilt),payload['room_dimensions'],out,data['lighting'])
            renderer=Silhouettes(rebuilt,out,camera,source/'sam2_masks.npz')
            actual,values,_=renderer.measure(out)
            report['rebuilt_iou']=actual
            report['export_passed']=actual>report['before_iou']+.001 and all(v>=o['before_iou']-.01 for v,o in zip(values,report['objects']))
        print(json.dumps({'job':source.name,**report}),flush=True)
