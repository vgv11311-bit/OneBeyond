"""Rebuild one saved scene from explicit, job-local photo review notes.

The notes are data for one capture, never global class rules or pasted pixels.
No provider call is made; the result is marked reviewed, not AI-verified.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path

from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.gemini_scene_engine import parse_scene


def apply_review(source: Path, notes_path: Path, output: Path, publish=False):
    source=source.resolve(); output=output.resolve()
    saved=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    notes=json.loads(notes_path.read_text(encoding='utf-8'))
    room=notes.get('estimated_room_dimensions',saved['room_dimensions'])
    if len(room)!=3 or any(float(value)<=0 for value in room):
        raise ValueError('Review room dimensions must be three positive estimates')
    specs=copy.deepcopy(saved['objects'])
    # Notes address the original saved inventory, so editing and removal must
    # use the same stable indices even when an earlier object is removed.
    for edit in notes.get('updates',[]):
        index=edit['index']
        if not 0<=index<len(specs): raise ValueError(f'Invalid review update: {index}')
        specs[index].update(edit['set'])
    for index in sorted(notes.get('remove_indices',[]),reverse=True):
        if not 0<=index<len(specs): raise ValueError(f'Invalid review removal: {index}')
        specs.pop(index)
    for item in notes.get('additions',[]):
        if item.get('source')!='reviewed_uploaded_photo':
            raise ValueError('Each added object needs explicit photo provenance.')
        specs.append(item)
    width,height=notes['source_image_size']
    candidates=[]
    for obj in specs:
        box=obj.get('image_bbox')
        if not box: continue
        candidate={k:obj[k] for k in ('label','center','size','yaw_degrees','confidence','asset_variant','wall_contact')}
        candidate['bbox']=[box[0]/width,box[1]/height,box[2]/width,box[3]/height]
        candidates.append(candidate)
    parse_scene({'room':list(room),'objects':candidates},(height,width,3))
    lighting=copy.deepcopy(metadata['metrics']['extra_details']['lighting'])
    lighting.setdefault('surface_colors',{}).update(notes.get('surface_colors',{}))
    output.mkdir(parents=True,exist_ok=True)
    result=BlenderSceneExporter.create_room_glb(str(output/'model.glb'),
        room,specs,lighting,preview_path=str(output/'preview.png'))
    if publish:
        state={key:metadata[key] for key in ('job_id','name','created_at','status') if key in metadata}
        state.update(job_id=output.name,name=notes['name'],status='completed',
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            glb_url=f'/api/spaces/{output.name}/model.glb',
            source_job_id=source.name,photo_review_status='manual_reviewed_estimate',
            review_notes=notes_path.name,
            metrics={
                'estimated_room_width':room[0],
                'estimated_room_length':room[1],
                'estimated_room_height':room[2],
                'total_frames_processed':metadata['metrics'].get('total_frames_processed',0),
                'processing_duration_sec':metadata['metrics'].get('processing_duration_sec',0),
                'engine_name':'Photo-reviewed Blender scene',
                'extra_details':{'scene_objects':specs,'lighting':lighting,
                                 'review_status':'manual_reviewed_estimate'},
            })
        (output/'metadata.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    return {'output':str(output),'objects':[o['label'] for o in specs],
            'published':publish,'export':result}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('notes',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    print(json.dumps(apply_review(args.source,args.notes,args.output,args.publish),indent=2))
