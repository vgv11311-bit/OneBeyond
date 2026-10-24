"""Independent photo construction review for any saved scene, then shared rebuild.

This is a provider-backed diagnostic replay, not a claim of fresh full-pipeline QA.
The saved upload is preserved; publication requires an explicit --publish.
"""
import argparse
import base64
import copy
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
from provider_credentials import groq_credentials
from reconstruction.groq_scene_engine import GroqSceneEngine
from reconstruction.provider_resilience import encode_overview
from refine_source_contract import refine


def review(source,output,publish=False,review_inputs=(),labels=None,offline=False,confirm_seating_contacts=False):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
    def identity(obj): return obj.get('label'),tuple(obj.get('image_bbox',[]))
    for previous in review_inputs:
        checked=json.loads(previous.read_text(encoding='utf-8'))
        by_identity={identity(o):o for o in checked['objects'] if o.get('appearance_reviewed')}
        for obj in payload['objects']:
            prior=by_identity.get(identity(obj))
            if prior:
                for key in ('object_appearance','appearance_source','appearance_reviewed','photo_components','asset_variant','construction_support','support_style','presence_review'):
                    if key in prior:
                        obj[key]=({**obj.get(key,{}),**prior[key]} if key=='object_appearance'
                                  else copy.deepcopy(prior[key]))
    if confirm_seating_contacts:
        # Explicit human feedback confirms existing relationships, never a
        # template, a new wall assignment, or a filename-specific override.
        for obj in payload['objects']:
            wall=obj.get('sectional_side') or obj.get('wall_contact')
            if obj.get('label')=='sofa' and wall in {'left','right','back'}:
                obj.update(wall_contact=wall,wall_confidence=1.,
                    wall_contact_source='user confirmed existing sofa-to-wall relationship')
                obj.pop('seating_fit_signature',None)
    image=cv2.imread(str(source/'extracted_frames'/'photo_0000.jpg'))
    if image is None: raise ValueError('Saved source photo is unavailable')
    ih,iw=image.shape[:2];data={'room':payload['room_dimensions'],'objects':[]}
    for spec in payload['objects']:
        obj=copy.deepcopy(spec)
        if not obj.get('image_bbox'): raise ValueError('Object has no source photo outline')
        obj['bbox']=[v/d for v,d in zip(obj['image_bbox'],[iw,ih,iw,ih])]
        if not offline and (labels is None or obj['label'] in labels): obj.pop('appearance_reviewed',None)
        else: obj['_skip_review']=True
        data['objects'].append(obj)
    engine=GroqSceneEngine();engine._analysis_deadline=time.monotonic()+300
    engine.analysis_status=lambda message:print(message,flush=True)
    _,model=groq_credentials()
    encoded=base64.b64encode(encode_overview(image)).decode('ascii')
    reviewed=data if offline else engine.describe_components(data,encoded,model,'')
    kept=[];rejected=[];specs=[]
    for i,obj in enumerate(reviewed['objects']):
        obj.pop('_skip_review',None)
        if obj.get('presence_review')=='independently_absent':
            rejected.append({'index':i,'label':obj['label'],'bbox':obj['bbox']});continue
        obj.pop('bbox',None);kept.append(i);specs.append(obj)
    if not specs: raise ValueError('Review rejected every object; original preserved')
    payload['objects']=specs
    path=output/'reviewed_scene_input.json';path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    mask_path=source/'sam2_masks.npz'
    if mask_path.exists() and len(kept)!=len(data['objects']):
        with np.load(mask_path,allow_pickle=False) as archive:
            if len(archive['masks'])!=len(data['objects']): raise ValueError('Saved instance masks do not match reviewed objects')
            values={key:(archive[key][kept] if key in {'masks','scores','labels','boxes'} else archive[key]) for key in archive.files}
        mask_path=output/'reviewed_masks.npz';np.savez_compressed(mask_path,**values)
    refine(source,output/'rebuilt',publish=publish,scene_input=path,mask_path_override=mask_path,reestimate_enclosure=True)
    report={'source':str(source),'rejected':rejected,'provider_calls':getattr(engine,'_groq_http_attempts',None),
            'duration_seconds':round(time.monotonic()-started,3),
            'reviewed':[{'label':o['label'],'appearance':o.get('object_appearance',{}),'verified':o.get('appearance_reviewed',False)} for o in specs]}
    (output/'appearance-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--review-input',action='append',type=Path,default=[],help='Merge independently reviewed fields by exact object identity, never geometry')
    parser.add_argument('--labels',nargs='+',help='Request only these construction labels')
    parser.add_argument('--offline',action='store_true',help='Rebuild supplied review evidence without a provider call')
    parser.add_argument('--confirm-seating-contacts',action='store_true',help='Apply explicit user confirmation of existing sofa wall relationships')
    args=parser.parse_args()
    os.environ['ONE_BEYOND_JOB_CHILD']='1'
    review(args.source,args.output,args.publish,args.review_input,args.labels,args.offline,args.confirm_seating_contacts)
