"""Shared quota-free refinement stages for any saved or fresh room upload."""
import copy
from .evidence_fusion import fuse_inventory
from .construction_contracts import refine_construction
from .joint_layout import refine_joint_layout
from .material_consensus import refine_materials


def refine_evidence_pipeline(specs,room,lighting,frame_paths,view_support,geometry_path,mask_path,measured_scale=False):
    lighting=copy.deepcopy(lighting)
    objects,inventory=fuse_inventory(specs,view_support,mask_path)
    objects,construction=refine_construction(objects)
    objects,dimensions,camera,layout=refine_joint_layout(objects,room,lighting.get('projection_camera'),geometry_path,mask_path,measured_scale)
    objects,materials=refine_materials(objects,frame_paths,view_support)
    if camera: lighting['projection_camera']=camera
    report=dict(inventory=inventory,construction=construction,joint_layout=layout,materials=materials,
        provider_calls=0,notice='Local evidence checks; unknown objects and hidden surfaces remain unresolved.')
    lighting['source_quality']=report
    return objects,dimensions,lighting,report


def mesh_alignment_report(renderer,specs):
    score,individual,canvas=renderer.measure(specs);objects=[]
    import numpy as np
    for index,value in zip(renderer.active,individual):
        target=renderer.targets[index];actual=canvas==index
        ty,tx=np.where(target);ay,ax=np.where(actual)
        centroid_error=(float(np.hypot(ax.mean()-tx.mean(),ay.mean()-ty.mean()))/max(canvas.shape)) if len(ax) and len(tx) else None
        objects.append(dict(index=index,label=specs[index]['label'],visible_iou=round(value,4),
            source_pixels=int(target.sum()),rendered_pixels=int(actual.sum()),
            normalized_centroid_error=centroid_error,needs_review=value<.35,
            missing_in_render=bool(target.sum()>=8 and actual.sum()==0)))
    return dict(method='source-camera triangle depth render',mean_visible_iou=round(score,4),objects=objects,
        notice='Visible mask agreement only; not a percentage of 3D accuracy or proof of correct identity/materials.')
