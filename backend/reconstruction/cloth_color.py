"""Photo-derived cloth colours, using instance masks when available."""
import copy
import cv2
import numpy as np
from scipy.signal import find_peaks

def color_cloth(specs, image, mask_path=None):
    result=copy.deepcopy(specs)
    masks=None
    if mask_path is not None and mask_path.exists():
        with np.load(mask_path,allow_pickle=False) as data:
            masks=data['masks'].copy()
    h,w=image.shape[:2]
    # Neutral room surfaces provide a conservative local illumination reference.
    # Exclude every detected object: do not normalize one cabinet to another.
    neutral=np.max(image,axis=2).astype(float)-np.min(image,axis=2)
    luminance=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
    wall_pixels=(neutral<24)&(luminance>90)&(luminance<240)
    for spec in specs:
        b=spec.get('image_bbox')
        if b:
            x1,y1,x2,y2=np.rint(b).astype(int)
            wall_pixels[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=False
    yy,xx=np.indices((h,w))
    for i,o in enumerate(result):
        if o.get('color_reviewed'):
            # A reviewed, source-measured material must survive offline replay.
            continue
        if o['label'] not in {'ceiling_fan','plant','bed','pillow','cabinet','blanket','duvet','throw','curtain','curtains','rug','book','bookshelf','wardrobe','dresser','wall_shelf','table','desk','side_table','nightstand','coffee_table','dining_table','chair','ergonomic_chair','office_chair','sofa','armchair','cup','bottle','vase','door','blinds','roller_shade','air_conditioner','cushion','keyboard','mouse','laptop','picture_frame','picture','wall_art'} or not o.get('image_bbox'): continue
        x1,y1,x2,y2=np.rint(o['image_bbox']).astype(int)
        x1,x2=np.clip([x1,x2],0,w); y1,y2=np.clip([y1,y2],0,h)
        if o['label']=='plant':
            for container in result:
                if container.get('label') not in {'vase','pot','plant_pot'} or not container.get('image_bbox'): continue
                vx1,vy1,vx2,vy2=container['image_bbox']
                if x1<=(vx1+vx2)/2<=x2 and y1+(y2-y1)*.35<vy1<y2:
                    # Foliage ends at its container; headboard/table pixels
                    # alongside the pot must not become leaf material.
                    y2=min(y2,round(vy1))
        if o['label']=='bed':
            # The upper band contains pillows/headboards, not the frame.
            # Sample the exposed lower structural rail and remove bedding below.
            y1=y1+round((y2-y1)*.72)
        if o['label'] in {'table','desk','coffee_table','dining_table'}:
            # The top is normally below the object's bbox top (chairs, lamps,
            # and objects on the top may extend higher). The old top 30% band
            # sampled those dark foreground objects instead of the tabletop.
            full_y1,full_h=y1,max(1,y2-y1)
            y1=full_y1+round(full_h*.18)
            y2=full_y1+max(round(full_h*.50),round(full_h*.18)+1)
        elif o['label'] in {'side_table','nightstand'}:
            y2=y1+max(1,round((y2-y1)*.30))
        if x2<=x1 or y2<=y1: continue
        patch=image[y1:y2,x1:x2]
        selected=np.ones(patch.shape[:2],dtype=bool)
        if masks is not None and i<len(masks):
            mask=cv2.resize(masks[i].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
            eroded=cv2.erode(mask,np.ones((3,3),np.uint8))
            selected=eroded[y1:y2,x1:x2]>0
            if selected.sum()<4:
                # Thin book spines can disappear under erosion. Keep their
                # original instance pixels; never replace them with background.
                selected=mask[y1:y2,x1:x2]>0
        # Nested foreground objects are not the supporting material: cushions
        # must not dye the rear pillow, nor decor recolor a desktop/headboard.
        for j, foreground in enumerate(result):
            if j == i or not foreground.get('image_bbox'): continue
            fx1,fy1,fx2,fy2=np.rint(foreground['image_bbox']).astype(int)
            if (o['label'] in {'table','desk','side_table','nightstand','coffee_table','dining_table'}
                    and foreground['label'] in {'chair','ergonomic_chair','office_chair'}):
                # A tucked-in chair is not the table's material even when its
                # box crosses too little of the table to count as nested.
                if fy2>y1 and fy1<y2 and fx2>x1 and fx1<x2:
                    selected[max(0,fy1-y1):max(0,min(y2,fy2)-y1),
                             max(0,fx1-x1):max(0,min(x2,fx2)-x1)]=False
            original=o['image_bbox']
            full_area=max(1,(original[2]-original[0])*(original[3]-original[1]))
            foreground_area=max(0,fx2-fx1)*max(0,fy2-fy1)
            intersection=max(0,min(fx2,original[2])-max(fx1,original[0]))*max(0,min(fy2,original[3])-max(fy1,original[1]))
            if 0 < foreground_area < full_area*.7 and intersection/foreground_area>.85:
                if masks is not None and j<len(masks):
                    foreground_mask=cv2.resize(masks[j].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
                    # A throw's rectangular bounding box also encloses visible
                    # duvet pixels. Exclude its silhouette, not that whole box.
                    selected &= foreground_mask[y1:y2,x1:x2]==0
                else:
                    selected[max(0,fy1-y1):max(0,min(y2,fy2)-y1),
                             max(0,fx1-x1):max(0,min(x2,fx2)-x1)]=False
        # Use the exposed instance after occluder removal. The strip below
        # another layer's bbox is often bed-frame fabric, not the duvet.
        o.pop('color_sampling', None)
        pixels=patch[selected]
        if len(pixels)<4:
            o['color_warning']='Insufficient visible object pixels; retained material'
            continue
        if o['label']=='bed' and o.get('asset_variant') in {'wooden','wooden_rails','wood'}:
            # A bed mask includes white sheets as well as its structural rail.
            # Separate the observed timber chroma from neutral bedding; retain
            # its measured hue, rather than assigning a stock wood palette.
            chroma=np.ptp(pixels.astype(float),axis=1)
            timber=pixels[chroma>=max(12.,float(np.percentile(chroma,60)))]
            if len(timber)>=16:
                light=cv2.cvtColor(timber.reshape(-1,1,3),cv2.COLOR_BGR2GRAY).ravel()
                lo,hi=np.percentile(light,[50,85])
                face=timber[(light>=lo)&(light<=hi)]
                pixels=face if len(face)>=16 else timber
                o['color_sampling']='exposed_frame_chroma'
        if o['label']=='plant':
            # SAM's plant mask can include a large amount of pale wall, while
            # nearby vases and leaves may share one detector box. Exclude the
            # independently detected vase and estimate the foliage from
            # chromatic pixels, not the mask's background-dominated median.
            before_exclusions=pixels
            for foreground in result:
                if foreground.get('label')!='vase' or not foreground.get('image_bbox'): continue
                fx1,fy1,fx2,fy2=np.rint(foreground['image_bbox']).astype(int)
                x0,x1c=max(x1,fx1),min(x2,fx2)
                y0,y1c=max(y1,fy1),min(y2,fy2)
                if x0<x1c and y0<y1c:
                    selected[max(0,y0-y1):max(0,y1c-y1),
                             max(0,x0-x1):max(0,x1c-x1)]=False
            foliage_pixels=patch[selected]
            if len(foliage_pixels)>=4:
                pixels=foliage_pixels
            else:
                pixels=before_exclusions
            if len(pixels)>=4:
                gray_patch=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(float)
                background=np.median(gray_patch,axis=1)[:,None]
                contrast_pixels=patch[selected & (gray_patch<background-30)]
                contrast_selection=12<=len(contrast_pixels)<len(pixels)*.5
                if contrast_selection:
                    pixels=contrast_pixels
                    o['color_sampling']='photo-derived sparse foliage contrast'
                hsv=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3)
                saturation=hsv[:,1].astype(float)
                foliage_luma=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2GRAY).reshape(-1)
                threshold,_=cv2.threshold(foliage_luma.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                dark=pixels[foliage_luma<=threshold]
                light=pixels[foliage_luma>threshold]
                # Thin dark leaves occupy far less than 18% of a detection.
                # Split their contrast from the dominant wall instead of selecting
                # a percentile that still mostly contains background pixels.
                if not contrast_selection and len(dark)>=12 and len(light)>len(dark)*2 and float(np.mean(foliage_luma[foliage_luma>threshold])-np.mean(foliage_luma[foliage_luma<=threshold]))>45:
                    pixels=dark
                    o['color_sampling']='photo-derived sparse foliage contrast'
                    hsv=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3)
                    saturation=hsv[:,1].astype(float)
                # Prefer the foliage's own color signal over a large number of
                # neutral wall pixels. The threshold adapts to each photo and
                # does not impose a stock plant hue or palette.
                chroma_floor=max(36.,float(np.percentile(saturation,80)))
                foliage=pixels if contrast_selection else pixels[saturation>=chroma_floor]
                if len(foliage)>=max(12,int(len(pixels)*.015)):
                    pixels=foliage
                    if not contrast_selection: o['color_sampling']='photo-derived foliage chroma'
                else:
                    foliage_luma=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2GRAY).reshape(-1)
                    cutoff=np.percentile(foliage_luma,18)
                    foliage=pixels[foliage_luma<=cutoff]
                    if len(foliage)>=4:
                        pixels=foliage
                    o['color_sampling']='photo-derived foliage contrast'
        if o['label']=='pillow' and len(pixels)>=16:
            # Seams and the shaded underside are not the broad fabric face.
            # Use observed mid-to-bright body pixels, not a white preset or
            # clipped highlights. Colored cushions retain their actual hue.
            body_light=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2GRAY).ravel()
            low,high=np.percentile(body_light,[55,90])
            body=pixels[(body_light>=low)&(body_light<=high)]
            if len(body)>=4:
                pixels=body
                o['color_sampling']='pillow_body_midtones'
        if o['label'] in {'sofa','armchair'} and len(pixels)>=32:
            # Upholstery masks contain deep creases, undersides and dark arms.
            # Those are lighting, not the broad exposed fabric's base colour.
            light=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2GRAY).ravel()
            low,high=np.percentile(light,[60,90])
            face=pixels[(light>=low)&(light<=high)]
            if len(face)>=16:
                pixels=face
                o['color_sampling']='upholstery_exposed_midtones'
        # Median suppresses highlights and small shadow regions; retain RGB.
        o['color']=np.rint(np.median(pixels[:,::-1],axis=0)).astype(int).tolist()
        o['color_source']='uploaded_photo_cloth_region' if o['label'] in {'blanket','duvet','throw'} else 'uploaded_photo_region'
        if o['label']=='cabinet' and patch.shape[1]>=40 and patch.shape[0]>=20:
            o.pop('recess_color',None)
            o.pop('recess_color_source',None)
            gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(float)
            profile=np.median(gray[int(len(gray)*.2):int(len(gray)*.85)],axis=0)
            trend=cv2.GaussianBlur(profile.reshape(1,-1),(0,0),3)[0]
            residual=profile-trend
            peaks,_=find_peaks(residual,height=5,prominence=5,distance=3)
            if len(peaks)>=8:
                gaps=np.diff(peaks)
                if np.std(gaps)/max(np.mean(gaps),1)<.5:
                    o['asset_variant']='slatted'
                    o['slat_count']=int(np.clip(len(peaks),8,64))
                    # Slat gaps are cast shadows, not the timber's base color.
                    # Use the observed raised faces instead of baking the gaps
                    # into every surface and then shading the geometry again.
                    # Brightest columns are often edge highlights. Promoting
                    # those to the entire timber material bleaches sunlit wood.
                    # Use positive-relief midtones, excluding highlight peaks.
                    face_columns=(residual>0)&(residual<=np.percentile(residual,75))
                    front_rows=np.zeros(selected.shape,dtype=bool)
                    front_rows[int(len(gray)*.2):int(len(gray)*.85)]=True
                    face_pixels=patch[selected & front_rows & face_columns[None,:]]
                    if len(face_pixels)>=16:
                        o['color']=np.rint(np.median(face_pixels[:,::-1],axis=0)).astype(int).tolist()
                        o['color_sampling']='slat_face_midtones'
                    # Preserve the observed recess contrast separately. A
                    # bright face material on the backing erases the grooves
                    # under browser ambient/environment illumination.
                    recess_columns=residual<=np.percentile(residual,25)
                    front_rows=np.zeros(selected.shape,dtype=bool)
                    front_rows[int(len(gray)*.2):int(len(gray)*.85)]=True
                    recess_pixels=patch[selected & front_rows & recess_columns[None,:]]
                    if len(recess_pixels)>=16:
                        recess=np.median(recess_pixels[:,::-1],axis=0)
                        if np.mean(recess)<np.mean(o['color'])*.85:
                            o['recess_color']=np.rint(recess).astype(int).tolist()
                            o['recess_color_source']='uploaded_photo_slat_recess'
        if o['label'] in {'cabinet','wall_shelf','vase','cup','bottle'} and wall_pixels.sum()>100:
            cx,cy=(x1+x2)/2,(y1+y2)/2
            nearby=wall_pixels & (((xx-cx)**2+(yy-cy)**2)<(h*.45)**2)
            if nearby.sum()>40:
                gain=float(np.clip(np.median(luminance[wall_pixels])/np.median(luminance[nearby]),.8,1.25))
                o['color']=np.rint(np.clip(np.array(o['color'])*gain,0,255)).astype(int).tolist()
                if o.get('recess_color_source')=='uploaded_photo_slat_recess':
                    o['recess_color']=np.rint(np.clip(np.array(o['recess_color'])*gain,0,255)).astype(int).tolist()
                o['illumination_gain']=round(gain,4)
    return result
