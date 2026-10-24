"""Recompile saved detection evidence into a new review; never overwrite a job."""
import json
import sys
import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.cloth_color import color_cloth
import cv2


def rebuild(source,refresh_colors=False,materials_only=False,wall_placement=False,fixture_colors=False,daylight=False,cabinet_colors=False,bedding=False,wardrobe_contact=False,structural=False,camera_frame=False,surfaces=False):
    source=Path(source).resolve()
    metadata=json.loads((source/'metadata.json').read_text())
    scene=json.loads((source/'blender_scene_input.json').read_text())
    extra=metadata['metrics']['extra_details']
    lighting=extra['lighting']
    from reconstruction.surface_evidence import bed_construction
    evidence=source if (source/'sam2_masks.npz').exists() else source.parent/extra.get('source_job_id',source.name)
    scene['objects']=bed_construction(scene['objects'],evidence/'sam2_masks.npz')
    if camera_frame:
        from reconstruction.room_frame import align_room
        from reconstruction.projection_fit import add_window_evidence
        evidence=source if (source/'moge_geometry.npz').exists() else source.parent/extra['source_job_id']
        # Replay pre-placement measurements, not an already fitted frontal room.
        diagnostics=extra.get('moge_geometry',{}).get('objects',[])
        for obj,measurement in zip(scene['objects'],diagnostics):
            if measurement.get('label')==obj['label'] and measurement.get('center_after'):
                obj['center']=measurement['center_after']
        scene['objects'],scene['room_dimensions'],lighting['projection_camera']=align_room(
            scene['objects'],scene['room_dimensions'],lighting.get('projection_camera'),
            evidence/'moge_geometry.npz',evidence/'sam2_masks.npz')
        scene['objects']=add_window_evidence(scene['objects'],evidence/'sam2_masks.npz',lighting['projection_camera'])
    if structural:
        from reconstruction.wall_contacts import resolve_bed_headboard,enforce_contacts
        from reconstruction.projection_fit import fit_scene
        scene['objects'],_=enforce_contacts(resolve_bed_headboard(scene['objects']),scene['room_dimensions'])
        scene['objects'],_=fit_scene(scene['objects'],scene['room_dimensions'],lighting.get('projection_camera'))
        scene['objects'],_=enforce_contacts(scene['objects'],scene['room_dimensions'])
    if wardrobe_contact:
        from reconstruction.wall_contacts import enforce_contacts
        scene['objects']=[enforce_contacts([obj],scene['room_dimensions'])[0][0]
                          if obj['label']=='wardrobe' else obj for obj in scene['objects']]
        from reconstruction.projection_fit import fit_wall_wardrobes
        scene['objects']=fit_wall_wardrobes(scene['objects'],scene['room_dimensions'],lighting.get('projection_camera'))
    if wall_placement:
        from reconstruction.wall_contacts import resolve_work_wall_from_projection,enforce_contacts
        from reconstruction.projection_fit import fit_chairs_to_fixed_tables
        evidence=source if (source/'moge_geometry.npz').exists() else source.parent/extra.get('source_job_id',source.name)
        scene['objects']=resolve_work_wall_from_projection(scene['objects'],scene['room_dimensions'],lighting.get('projection_camera'),evidence/'moge_geometry.npz')
        scene['objects'],_=enforce_contacts(scene['objects'],scene['room_dimensions'])
        scene['objects'],_=fit_chairs_to_fixed_tables(scene['objects'],scene['room_dimensions'],lighting.get('projection_camera'))
    if materials_only:
        prepared,adjustments=scene['objects'],[]
    else:
        prepared,adjustments=prepare_scene_specs(scene['objects'],scene['room_dimensions'],
                                                camera=lighting.get('projection_camera'))
    # This replay is for bedding/support geometry, not a new furniture fit.
    labels={'bed','pillow','cushion','blanket','duvet','throw','wall_shelf','book','vase','bottle','cup','picture','wall_art','picture_frame'}
    objects=[new if structural or old['label'] in labels else old for old,new in zip(scene['objects'],prepared)]
    if bedding:
        from reconstruction.bedding_appearance import add_fabric_patterns,order_pillows
        evidence=source if (source/'extracted_frames').exists() else source.parent/extra['source_job_id']
        paths=list((evidence/'extracted_frames').glob('photo_*.jpg'))
        if not paths: raise ValueError('No source photo for fabric patterns')
        objects=add_fabric_patterns(order_pillows(objects),cv2.imread(str(paths[0])),evidence/'sam2_masks.npz')
    if daylight:
        from reconstruction.daylight import estimate_daylight
        evidence=source if (source/'extracted_frames').exists() else source.parent/extra['source_job_id']
        paths=list((evidence/'extracted_frames').glob('photo_*.jpg'))
        if not paths: raise ValueError('No source photo for daylight analysis')
        lighting['daylight']=estimate_daylight(cv2.imread(str(paths[0])),objects,
                                              scene['room_dimensions'],lighting.get('projection_camera'))
    if fixture_colors:
        from reconstruction.fixture_lighting import add_fixture_lighting
        evidence=source if (source/'extracted_frames').exists() else source.parent/extra['source_job_id']
        paths=list((evidence/'extracted_frames').glob('photo_*.jpg'))
        if not paths: raise ValueError('No source photo for fixture colors')
        objects=add_fixture_lighting(objects,cv2.imread(str(paths[0])))
    if refresh_colors or cabinet_colors:
        evidence=source
        if not (evidence/'sam2_masks.npz').exists() and extra.get('source_job_id'):
            evidence=source.parent/extra['source_job_id']
        paths=list((evidence/'extracted_frames').glob('photo_*.jpg'))
        if not paths: raise ValueError('No source photo available for a color review')
        image=cv2.imread(str(paths[0]))
        if image is None: raise ValueError('Unreadable source photo')
        sampled=color_cloth(objects,image,evidence/'sam2_masks.npz')
        if cabinet_colors:
            # Keep all geometry and other materials intact during a timber review.
            for old,new in zip(objects,sampled):
                if old['label']=='cabinet':
                    for key in ('color','color_source','color_sampling','illumination_gain'):
                        if key in new: old[key]=new[key]
        else:
            objects=sampled
    if surfaces:
        from reconstruction.surface_evidence import floor_material,window_colors
        evidence=source if (source/'extracted_frames').exists() else source.parent/extra['source_job_id']
        image=cv2.imread(str(next((evidence/'extracted_frames').glob('photo_*.jpg'))))
        lighting['floor_material']=floor_material(image,objects)
        objects=window_colors(objects,image)
    job_id=str(uuid.uuid4())[:12]
    destination=source.parent/job_id
    BlenderSceneExporter.create_room_glb(str(destination/'model.glb'),scene['room_dimensions'],objects,lighting)
    metadata.update(job_id=job_id,name=('Color-space corrected - ' if materials_only else 'Photo color review - ' if refresh_colors else 'Detail geometry review - ')+metadata['name'],
                    created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    glb_url=f'/api/spaces/{job_id}/model.glb',stage='Detail geometry review ready',
                    logs=['Recompiled saved upload evidence, without new detection or overwriting the source job.'])
    if fixture_colors: metadata['name']='Fixture shades - '+metadata['name'].removeprefix('Color-space corrected - ')
    if daylight: metadata['name']='Window lighting review - '+metadata['name'].removeprefix('Color-space corrected - ')
    if cabinet_colors: metadata['name']='Cabinet color review - '+metadata['name'].removeprefix('Color-space corrected - ')
    if bedding: metadata['name']='Lighting and bedding review - '+metadata['name'].removeprefix('Window lighting review - ').removeprefix('Color-space corrected - ')
    if wardrobe_contact: metadata['name']='Fabric and cupboard review - '+source.name
    if structural: metadata['name']='Saved-evidence regression review - '+metadata['name'].removeprefix('Lighting and bedding review - ')
    if camera_frame or surfaces: metadata['name']='Saved-evidence camera and materials review - '+source.name
    extra.update(scene_objects=objects,source_job_id=source.name,
                 detail_replay_adjustments=adjustments,
                 self_correction={'status':'not_rerun','reason':'Saved-evidence detail-only replay'})
    (destination/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(job_id)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--refresh-colors',action='store_true')
    parser.add_argument('--materials-only',action='store_true',help='Preserve all saved object dimensions and poses')
    parser.add_argument('--wall-placement',action='store_true',help='Recheck work-desk contacts against the saved camera')
    parser.add_argument('--fixture-colors',action='store_true',help='Resample fixture chromaticity from saved photo evidence')
    parser.add_argument('--daylight',action='store_true',help='Analyze window lighting using saved photo evidence')
    parser.add_argument('--cabinet-colors',action='store_true',help='Resample only cabinet timber; preserve lighting and geometry')
    parser.add_argument('--bedding',action='store_true',help='Extract fabric design and resolve visible cushion layers')
    parser.add_argument('--wardrobe-contact',action='store_true',help='Normalize cupboard local axes and wall contact')
    parser.add_argument('--structural',action='store_true',help='Recheck generic headboard evidence and physical contacts')
    parser.add_argument('--camera-frame',action='store_true')
    parser.add_argument('--surfaces',action='store_true')
    args=parser.parse_args()
    rebuild(args.source,args.refresh_colors,args.materials_only,args.wall_placement,args.fixture_colors,args.daylight,args.cabinet_colors,args.bedding,args.wardrobe_contact,args.structural,args.camera_frame,args.surfaces)
