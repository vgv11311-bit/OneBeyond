"""Headless Blender scene compiler for One Beyond.

This file is executed by the dedicated Python 3.13 + bpy environment.  The
Flask process deliberately does not import bpy because the rest of the backend
continues to run on Python 3.10.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import bpy


MATERIALS: Dict[str, Any] = {}
TEXTURES: Dict[str, Any] = {}


def ensure_texture_uvs(mesh):
    """Give custom textured meshes a valid UV channel, preserving authored UVs.

    Procedural corner/back meshes do not receive Blender primitive UVs. The
    glTF exporter otherwise emits texCoord=-1 for their image materials.
    Use face-oriented box projection, never source-image projection.
    """
    if mesh.uv_layers or not mesh.vertices: return
    if not any(mat and mat.use_nodes and any(n.type=='TEX_IMAGE' for n in mat.node_tree.nodes)
               for mat in mesh.materials): return
    coords=[v.co for v in mesh.vertices]
    low=[min(v[a] for v in coords) for a in range(3)]
    span=[max(max(v[a] for v in coords)-low[a],1e-6) for a in range(3)]
    uv=mesh.uv_layers.new(name='GeneratedFinishUV')
    for polygon in mesh.polygons:
        normal_axis=max(range(3),key=lambda a:abs(polygon.normal[a]))
        axes=[a for a in range(3) if a!=normal_axis]
        for index in polygon.loop_indices:
            point=mesh.vertices[mesh.loops[index].vertex_index].co
            uv.data[index].uv=tuple((point[a]-low[a])/span[a] for a in axes)


def grid_uv(mesh, columns, rows):
    uv=mesh.uv_layers.new(name='FabricUV')
    layer=(columns+1)*(rows+1)
    for loop in mesh.loops:
        index=loop.vertex_index%layer
        uv.data[loop.index].uv=(index%(columns+1)/columns,1-index//(columns+1)/rows)


def fabric_pattern_material(spec, fallback):
    if spec.get('fabric_pattern_source')!='photo_measured_weave': return fallback
    import base64
    import tempfile
    with tempfile.TemporaryDirectory(prefix='one-beyond-fabric-') as directory:
        path=Path(directory)/'pattern.png'
        path.write_bytes(base64.b64decode(spec['fabric_pattern_png'],validate=True))
        texture=bpy.data.images.load(str(path),check_existing=False)
        texture.colorspace_settings.name='sRGB'
        texture.pack()
    mat=fallback.copy()
    mat.name='Photo fabric pattern'
    nodes=mat.node_tree.nodes
    principled=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    image_node=nodes.new('ShaderNodeTexImage')
    image_node.image=texture
    image_node.interpolation='Linear'
    # A synthetic weave normal on top of an observed print changes its motif
    # and creates shimmering highlights. Cloth geometry already supplies folds.
    for link in list(principled.inputs['Normal'].links):
        mat.node_tree.links.remove(link)
    mat.node_tree.links.new(image_node.outputs['Color'],principled.inputs['Base Color'])
    return mat


def procedural_wallpaper_material(color, fallback, pattern=None, measured=None):
    """Render a repeatable finish from measured color, never a photo cut-out.

    The image is generated entirely from periodic low-frequency functions, so
    tiling cannot reveal a rectangular source-photo boundary or embedded decor.
    """
    from photo_materials import photo_linear_rgba
    import numpy as np
    base=np.asarray(photo_linear_rgba(color)[:3],dtype=np.float32)
    y,x=np.mgrid[0:128,0:128].astype(np.float32)
    x*=2*math.pi/128; y*=2*math.pi/128
    relief=(.42*np.sin(3*x+2*np.sin(2*y))
            +.28*np.cos(5*y-2*x)+.18*np.sin(9*x+7*y))
    if pattern=='woven_wave':
        relief=(np.cos(12*x+1.3*np.sin(2*y))**12
                +np.cos(16*y+1.1*np.sin(2*x))**12)-.45
    relief-=float(relief.mean())
    variation=relief*.075
    if measured and measured.get('source')=='measured_periodic_statistics':
        from finish_pattern import synthesize_pattern
        variation=synthesize_pattern(measured)
    rgb=np.clip(base[None,None,:]*(1+variation[:,:,None]),0,1)
    # glTF base-color images are sRGB, unlike the linear shader constant.
    # Encoding linear values as image bytes darkens the finish a second time.
    rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
    pixels=np.concatenate((rgb,np.ones((128,128,1),dtype=np.float32)),axis=2)
    image=bpy.data.images.new('Procedural wallpaper finish',width=128,height=128,alpha=True)
    image.pixels.foreach_set(pixels.astype(np.float32).ravel())
    image.pack()
    mat=fallback.copy(); mat.name='Measured-color procedural wallpaper'
    nodes=mat.node_tree.nodes
    shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    texture=nodes.new('ShaderNodeTexImage'); texture.image=image
    texture.extension='REPEAT'; texture.interpolation='Linear'
    mat.node_tree.links.new(texture.outputs['Color'],shader.inputs['Base Color'])
    return mat


def fixture_diffuser_material(spec, fallback):
    from photo_materials import photo_linear_rgba
    light=spec.get('fixture_light') or {}
    intensity=float(light.get('intensity',0))
    if intensity<=0: return fallback
    mat=fallback.copy(); mat.name='Photo luminous diffuser'
    shader=mat.node_tree.nodes.get('Principled BSDF')
    color=photo_linear_rgba([255*v for v in light.get('color',[1,1,1])])
    shader.inputs['Emission Color'].default_value=color
    shader.inputs['Emission Strength'].default_value=min(4.,max(1.,intensity*.6))
    return mat


def build_wall_finishes(surfaces,room,parent,objects):
    """Real wall-mounted finish volumes and frame rails from validated evidence."""
    from photo_materials import photo_linear_rgba
    from wall_appearance import subtract_openings, visible_finish_parts
    width,length,height=room
    for i,surface in enumerate(surfaces):
        wall=surface['wall']; span=width if wall=='back' else length
        finish_type=surface.get('kind','paint')
        finish_surface={'wood':'wood','fabric':'fabric','paint':'wall','stone':'wall'}.get(surface.get('finish_material'))
        if finish_type=='paint': finish_surface='wall'
        finish_roughness=.72 if finish_type=='headboard' else .86
        mat=material(f'Observed wall finish {i}',photo_linear_rgba(surface['color']),
                     roughness=finish_roughness,surface=finish_surface)
        if finish_type=='wallpaper':
            mat=procedural_wallpaper_material(surface['color'],mat,surface.get('pattern'),surface.get('procedural_pattern'))
        def piece(name,bounds,depth,material_value,texture_bounds=None):
            u0,v0,u1,v1=bounds
            u=(u0+u1)/2; y=(v0+v1)*height/2
            if wall=='back':
                dims=[(u1-u0)*span,depth,(v1-v0)*height]
                center=[(u-.5)*width,y,-length+.15+.045+depth/2]
            else:
                dims=[depth,(u1-u0)*span,(v1-v0)*height]
                z=(-u*length+.15) if wall=='left' else (-length+u*length+.15)
                center=[(-1 if wall=='left' else 1)*(width/2-.045-depth/2),y,z]
            obj=add_box(name,dims,center,material_value,parent=parent)
            if finish_surface=='wood':
                return obj  # Retain metric grain UVs; do not stretch one tile across the wall.
            uv=obj.data.uv_layers.active
            for loop in obj.data.loops:
                p=obj.data.vertices[loop.vertex_index].co
                horizontal=p.x/dims[0] if wall=='back' else (1 if wall=='left' else -1)*p.y/dims[1]
                uu=horizontal+.5; vv=p.z/dims[2]+.5
                if finish_type=='wallpaper':
                    uu=(u0+uu*(u1-u0))*span*.75
                    vv=(v0+vv*(v1-v0))*height*.75
                elif texture_bounds:
                    a,b,c,d=texture_bounds
                    uu=(u0+uu*(u1-u0)-a)/(c-a); vv=(v0+vv*(v1-v0)-b)/(d-b)
                uv.data[loop.index].uv=(uu,vv)
            return obj
        openings=[]
        for obj in objects:
            if obj.get('label') not in {'window','door'} or (obj.get('wall') or obj.get('wall_contact'))!=wall: continue
            x,y,z=obj['center']; sw,sd,sh=obj['size']
            u=x/width+.5 if wall=='back' else ((.15-z)/length if wall=='left' else (z+length-.15)/length)
            openings.append([u-sw/span/2,(y-sh/2)/height,u+sw/span/2,(y+sh/2)/height])
        finish_depth=.045 if finish_type=='headboard' else .009
        for k,part in enumerate(visible_finish_parts(surfaces,i,openings)):
            piece(f'Observed_{wall}_{surface["kind"]}_{i}_{k}',part,finish_depth,mat,surface['bounds'])
        # Thin raised rails represent observed panel frames; no default count.
        for j,b in enumerate(surface.get('frame_boxes',[])):
            u0,v0,u1,v1=b; du=min(.025/span,(u1-u0)/6); dv=min(.025/height,(v1-v0)/6)
            for k,rail in enumerate(([u0,v0,u0+du,v1],[u1-du,v0,u1,v1],
                                      [u0+du,v0,u1-du,v0+dv],[u0+du,v1-dv,u1-du,v1])):
                for fragment,part in enumerate(subtract_openings(rail,openings)):
                    piece(f'Observed_Panel_Rail_{i}_{j}_{k}_{fragment}',part,.025,mat)


def _blender_point(point: Sequence[float]) -> tuple[float, float, float]:
    """Map the app's Y-up coordinates to Blender's Z-up coordinates."""
    x, y_up, z_depth = (float(value) for value in point)
    return x, -z_depth, y_up


def _rgba(value: Sequence[float] | None, fallback: Sequence[int]) -> tuple[float, float, float, float]:
    source = list(value or fallback)[:4]
    while len(source) < 4:
        source.append(255)
    if max(source) > 1:
        source = [channel / 255.0 for channel in source]
    return tuple(max(0.0, min(1.0, float(channel))) for channel in source)


def surface_texture(name: str, color: Sequence[float], surface: str, *, normal: bool = False):
    """Create a small embedded PBR texture without external files or licences."""
    key = f"{name}:{surface}:{normal}:{tuple(round(float(v), 3) for v in color)}"
    if key in TEXTURES:
        return TEXTURES[key]
    size = 256 if surface in {'carpet','tile','wood'} else 64
    image = bpy.data.images.new(name=key, width=size, height=size, alpha=True)
    pixels = []
    rgba = _rgba(color, [180, 180, 180, 255])
    for y in range(size):
        for x in range(size):
            u, v = x / size, y / size
            if surface == "wood":
                bend=math.sin(u*math.tau)*.8+math.sin(u*math.tau*3)*.2
                wave=math.sin(v*math.tau*53+bend)
                fine=(math.sin(v*math.tau*89+bend*1.7)+
                      math.sin(v*math.tau*121-bend*.6))*.5
                # Fine irregular grain, not broad decorative stripes. Mean
                # color stays at the measured source material, with no crop.
                variation=wave*.028+fine*.018
                nx,ny=.5,.5+wave*.012+fine*.006
            elif surface == "fabric":
                weave_x = math.sin(u * math.pi * 42)
                weave_y = math.sin(v * math.pi * 42)
                variation = (weave_x + weave_y) * 0.012 + weave_x * weave_y * 0.018
                nx, ny = 0.5 + weave_x * 0.025, 0.5 + weave_y * 0.025
            elif surface == 'carpet':
                grain=math.sin(x*127.1+y*311.7)*43758.5453
                grain=(grain-math.floor(grain))-.5
                variation=grain*.28
                nx,ny=.5+grain*.18,.5-grain*.13
            elif surface == 'tile':
                grout=x<2 or y<2
                grain=math.sin((u*23+v*31)*math.pi)*math.sin((u*17-v*29)*math.pi)
                variation=-.14 if grout else grain*.015
                nx,ny=.5,.5
            elif surface == "wall":
                grain = math.sin((u * 37 + v * 61) * math.pi) * math.sin((u * 19 - v * 29) * math.pi)
                variation = grain * 0.012
                nx, ny = 0.5 + grain * 0.008, 0.5 - grain * 0.008
            else:
                grain = math.sin((u * 23 + v * 31) * math.pi)
                variation = grain * 0.008
                nx, ny = 0.5 + grain * 0.006, 0.5
            if normal:
                pixels.extend((nx, ny, 1.0, 1.0))
            else:
                factor = max(0.84, min(1.12, 1.0 + variation))
                # PNG/glTF base-color texture bytes are sRGB, while material
                # inputs above are linear. Writing linear values as sRGB bytes
                # applies the transfer function twice in the browser (too dark).
                linear=[min(1.,max(0.,channel*factor)) for channel in rgba[:3]]
                encoded=[12.92*v if v<=.0031308 else 1.055*v**(1/2.4)-.055 for v in linear]
                pixels.extend((*encoded,rgba[3]))
    image.pixels.foreach_set(pixels)
    image.pack()
    if normal:
        image.colorspace_settings.name = "Non-Color"
    TEXTURES[key] = image
    return image


def material(
    name: str,
    color: Sequence[float],
    *,
    roughness: float = 0.55,
    metallic: float = 0.0,
    alpha: float = 1.0,
    surface: str | None = None,
):
    key = f"{name}:{tuple(round(float(v), 4) for v in color)}:{roughness}:{metallic}:{alpha}:{surface}"
    if key in MATERIALS:
        return MATERIALS[key]
    mat = bpy.data.materials.new(name=name)
    mat['procedural_surface']=surface or ''
    rgba = _rgba(color, [190, 190, 190, 255])
    mat.diffuse_color = (rgba[0], rgba[1], rgba[2], alpha)
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], alpha)
        principled.inputs["Roughness"].default_value = roughness
        principled.inputs["Metallic"].default_value = metallic
        if "Alpha" in principled.inputs:
            principled.inputs["Alpha"].default_value = alpha
        if surface:
            base = mat.node_tree.nodes.new("ShaderNodeTexImage")
            base.name = f"{surface.title()} Base Color"
            base.image = surface_texture(name, rgba, surface)
            normal_texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
            normal_texture.name = f"{surface.title()} Normal"
            normal_texture.image = surface_texture(name, rgba, surface, normal=True)
            normal_map = mat.node_tree.nodes.new("ShaderNodeNormalMap")
            normal_map.inputs["Strength"].default_value = 0.24 if surface == "fabric" else 0.16
            mat.node_tree.links.new(base.outputs["Color"], principled.inputs["Base Color"])
            mat.node_tree.links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
            mat.node_tree.links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
    if alpha < 1:
        mat.surface_render_method = "DITHERED"
    MATERIALS[key] = mat
    return mat


def add_empty(name: str, center: Sequence[float], yaw_degrees: float = 0.0, parent=None):
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    empty.empty_display_type = "PLAIN_AXES"
    empty.location = _blender_point(center)
    empty.rotation_euler[2] = math.radians(float(yaw_degrees))
    empty.parent = parent
    return empty


def add_box(
    name: str,
    size: Sequence[float],
    local_center: Sequence[float],
    mat,
    *,
    parent=None,
    bevel: float = 0.0,
    bevel_segments: int = 4,
):
    width, depth, height = (max(0.006, float(value)) for value in size)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = (width, depth, height)
    obj.location = _blender_point(local_center)
    obj.parent = parent
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat.get('procedural_surface')=='wood':
        # The stock cube UV atlas stretched the same grain differently on a
        # four-metre wall panel and a small drawer top. Use metres on each face.
        uv=obj.data.uv_layers.active
        for face in obj.data.polygons:
            axis=max(range(3),key=lambda k:abs(face.normal[k]))
            axes=(0,1) if axis==2 else ((0,2) if axis==1 else (1,2))
            for index in face.loop_indices:
                p=obj.data.vertices[obj.data.loops[index].vertex_index].co
                uv.data[index].uv=(p[axes[0]],p[axes[1]])
    if bevel > 0:
        modifier = obj.modifiers.new(name="Controlled edge bevel", type="BEVEL")
        modifier.width = min(float(bevel), min(width, depth, height) * 0.22)
        modifier.segments = max(2, int(bevel_segments))
        modifier.limit_method = "ANGLE"
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.materials.append(mat)
    return obj


def add_cylinder(
    name: str,
    radius: float,
    height: float,
    local_center: Sequence[float],
    mat,
    *,
    parent=None,
    vertices: int = 48,
):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=max(0.006, float(radius)),
        depth=max(0.006, float(height)),
        location=_blender_point(local_center),
    )
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.data.materials.append(mat)
    # Smooth only the curved barrel. Keeping the caps flat avoids the
    # balloon-like shading that full-object smooth shading creates.
    for polygon in obj.data.polygons:
        polygon.use_smooth = abs(polygon.normal.z) < 0.9
    return obj


def add_cone(
    name: str,
    radius_bottom: float,
    radius_top: float,
    height: float,
    local_center: Sequence[float],
    mat,
    *,
    parent=None,
):
    bpy.ops.mesh.primitive_cone_add(
        vertices=48,
        radius1=max(0.006, float(radius_bottom)),
        radius2=max(0.006, float(radius_top)),
        depth=max(0.006, float(height)),
        location=_blender_point(local_center),
    )
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = abs(polygon.normal.z) < 0.9
    return obj


def _four_legs(parent, width: float, depth: float, bottom: float, top: float, mat, prefix: str):
    leg = max(0.035, min(width, depth) * 0.075)
    height = max(0.05, top - bottom)
    for index, (sx, sz) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        add_box(
            f"{prefix}_Leg_{index}",
            [leg, leg, height],
            [sx * (width / 2 - leg), bottom + height / 2, sz * (depth / 2 - leg)],
            mat,
            parent=parent,
            bevel=leg * 0.12,
        )


def build_bed(parent, size: Sequence[float], mats: Dict[str, Any], variant='default', headboard=True):
    width, depth, height = size
    bottom = -height / 2
    from bed_dimensions import bed_layers
    frame_h,mattress_h=bed_layers(height,variant)
    # The mattress bottom is exactly the frame top. The previous percentage
    # placement left a visible air gap for taller inferred bed bounds.
    mattress_center = bottom + frame_h + mattress_h / 2
    if variant in {'wooden_rails','painted_slatted'}:
        # Keep the established mattress support height; expose the space below
        # the rails instead of replacing a legged frame with a solid plinth.
        rail_h=frame_h*.48
        for x in (-width*.46,width*.46):
            add_box('Bed_Side_Rail',[width*.08,depth,rail_h],
                    [x,bottom+frame_h-rail_h/2,0],mats['wood'],parent=parent,bevel=.008)
        for z in (-depth*.46,depth*.46):
            add_box('Bed_End_Rail',[width,depth*.08,rail_h],
                    [0,bottom+frame_h-rail_h/2,z],mats['wood'],parent=parent,bevel=.008)
        _four_legs(parent,width*.9,depth*.9,bottom,bottom+frame_h,mats['wood'],'Bed')
    else:
        add_box("Bed_Frame", [width, depth, frame_h], [0, bottom + frame_h / 2, 0], mats["wood"], parent=parent, bevel=0.025)
    add_box(
        "Bed_Mattress",
        [width * 0.95, depth * 0.92, mattress_h],
        [0, mattress_center, depth * 0.015],
        mats["fabric_light"],
        parent=parent,
        bevel=min(0.035, mattress_h * 0.14),
        bevel_segments=4,
    )
    if not headboard:
        return
    headboard_depth = min(0.12, depth * 0.07)
    if variant=='painted_slatted':
        z=-depth/2+headboard_depth/2
        rail=.065
        for x in (-width/2+rail/2,width/2-rail/2):
            add_box('Bed_Head_Post',[rail,headboard_depth,height],[x,0,z],mats['wood'],parent=parent,bevel=.006)
        add_box('Bed_Head_Top',[width,headboard_depth,rail],[0,height/2-rail/2,z],mats['wood'],parent=parent,bevel=.006)
        lower=bottom+frame_h+mattress_h*.7
        slat_h=max(.04,height/2-rail-lower)
        count=max(4,round(width/.15))
        for i in range(count):
            add_box('Bed_Head_Slat',[width/(count+1)*.35,headboard_depth*.5,slat_h],
                    [-width/2+(i+1)*width/(count+1),lower+slat_h/2,z],mats['wood'],parent=parent,bevel=.003)
        foot_bottom=bottom+frame_h*.48
        foot_h=frame_h+mattress_h+.04-(foot_bottom-bottom)
        add_box('Bed_Foot_Panel',[width,headboard_depth,foot_h],
                [0,foot_bottom+foot_h/2,depth/2-headboard_depth/2],mats['wood'],parent=parent,bevel=.008)
        return
    add_box(
        "Bed_Headboard",
        [width, headboard_depth, height],
        [0, 0, -depth / 2 + headboard_depth / 2],
        mats["wood"],
        parent=parent,
        bevel=0.028,
    )


def build_pillow(parent, size: Sequence[float], mats: Dict[str, Any],pose=None):
    width, depth, height = size
    vertices, faces = [], []
    n = 24
    layer = (n+1)*(n+1)
    for side in (1,-1):
        for row in range(n+1):
            v = 2*row/n-1
            for col in range(n+1):
                u = 2*col/n-1
                # Taper corners and flatten the sewn perimeter. Broad filled
                # faces retain a rectangular pillow silhouette.
                x = width/2*u*(1-.06*abs(v)**6)
                z = depth/2*v*(1-.06*abs(u)**6)
                fill = max(0,(1-abs(u)**8)*(1-abs(v)**8))**.5
                wrinkle = .012*math.sin(u*21+v*4)*fill*(1-fill)
                y = side*height/2*(.08+.92*fill+wrinkle)
                if pose=='upright' or (pose is None and height>=depth*.75):
                    # Upright headboard pillow: broad face lies in XY, not XZ.
                    # A rotation, not an axis reflection: preserve outward
                    # winding so single-sided GLTF materials remain visible.
                    vertices.append(_blender_point([x,z/depth*height,-y/height*depth]))
                else:
                    vertices.append(_blender_point([x,y,z]))
    for row in range(n):
        for col in range(n):
            a=row*(n+1)+col
            faces.append((a,a+n+1,a+n+2,a+1))
            faces.append((a+layer,a+1+layer,a+n+2+layer,a+n+1+layer))
    perimeter = (list(range(n+1)) +
                 [r*(n+1)+n for r in range(1,n+1)] +
                 [n*(n+1)+c for c in range(n-1,-1,-1)] +
                 [r*(n+1) for r in range(n-1,0,-1)])
    for index,a in enumerate(perimeter):
        b=perimeter[(index+1)%len(perimeter)]
        faces.append((a,b,b+layer,a+layer))
    mesh=bpy.data.meshes.new("Pillow_Closed_Cushion")
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    obj=bpy.data.objects.new("Pillow_Cushion",mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent=parent
    obj.data.materials.append(mats["fabric_light"])
    grid_uv(mesh,n,n)
    for polygon in mesh.polygons:
        polygon.use_smooth=True


def build_table(parent, size: Sequence[float], mats: Dict[str, Any], side_table: bool = False, variant='default'):
    width, depth, height = size
    bottom = -height / 2
    top_h = min(0.11, max(0.055, height * 0.14))
    add_box("Table_Top", [width, depth, top_h], [0, height / 2 - top_h / 2, 0], mats["wood"], parent=parent, bevel=0.018)
    if side_table and variant=='open_shelf':
        panel=min(.045,width*.08)
        for x in (-width/2+panel/2,width/2-panel/2):
            add_box('Bedside_Side',[panel,depth,height-top_h],[x,-top_h/2,0],mats['wood'],parent=parent,bevel=.005)
        for y in (bottom+panel/2,bottom+height*.42):
            add_box('Bedside_Shelf',[width-panel*2,depth,panel],[0,y,0],mats['wood'],parent=parent,bevel=.005)
        return
    if side_table and height > 0.35:
        cabinet_h = height * 0.38
        add_box("Side_Table_Cabinet", [width * 0.82, depth * 0.82, cabinet_h], [0, height / 2 - top_h - cabinet_h / 2, 0], mats["wood_dark"], parent=parent, bevel=0.022)
    _four_legs(parent, width * 0.86, depth * 0.82, bottom, height / 2 - top_h, mats["wood_dark"], "Table")


def build_round_table(parent,size,mats,frame_color=None):
    w,d,h=size
    radius=min(w,d)*.5
    from photo_materials import photo_linear_rgba
    frame=material('Photo round table frame',photo_linear_rgba(frame_color),roughness=.38,metallic=.6) if frame_color else mats['metal']
    add_cylinder('Round_Table_Top',radius,h*.065,[0,h*.4675,0],mats['fabric_light'],parent=parent,vertices=64)
    leg_radius=max(.008,min(.018,radius*.065))
    for index in range(3):
        angle=index*math.tau/3
        x,z=math.cos(angle)*radius*.72,math.sin(angle)*radius*.72
        lower=-h*.43
        upper=h*.435+.002  # overlap the underside, not a disconnected leg
        add_cylinder(f'Round_Table_Leg_{index}',leg_radius,upper-lower,[x,(upper+lower)/2,z],frame,parent=parent,vertices=16)
    bpy.ops.mesh.primitive_torus_add(major_segments=64,minor_segments=8,
        location=_blender_point([0,-h*.43,0]),major_radius=radius*.73,
        minor_radius=leg_radius*.8)
    ring=bpy.context.object; ring.name='Round_Table_Base_Ring'; ring.parent=parent
    ring.data.materials.append(frame)


def build_chair(parent, size: Sequence[float], mats: Dict[str, Any]):
    width, depth, height = size
    bottom = -height / 2
    seat_thickness = min(0.12, max(0.07, height * 0.12))
    seat_y = bottom + height * 0.52
    add_box("Chair_Seat", [width, depth * 0.86, seat_thickness], [0, seat_y, depth * 0.05], mats["fabric"], parent=parent, bevel=0.014, bevel_segments=3)
    back_h = height * 0.43
    back_d = min(0.12, depth * 0.18)
    add_box("Chair_Back", [width * 0.96, back_d, back_h], [0, height / 2 - back_h / 2, -depth / 2 + back_d / 2], mats["fabric"], parent=parent, bevel=0.014, bevel_segments=3)
    _four_legs(parent, width * 0.8, depth * 0.70, bottom, seat_y - seat_thickness / 2, mats["metal"], "Chair")


def build_ergonomic_chair(parent, size, mats):
    width, depth, height = size
    bottom = -height / 2
    seat_y = bottom + height * .43
    add_box("Ergonomic_Seat", [width*.78, depth*.70, height*.07],
            [0, seat_y, depth*.07], mats["fabric"], parent=parent,
            bevel=min(.025, height*.018), bevel_segments=6)
    add_cylinder("Chair_Lift", width*.045, height*.37,
                 [0, bottom+height*.22, 0], mats["metal"], parent=parent)
    # A contoured back with lumbar curvature, narrowing toward the shoulders.
    vertices, faces = [], []
    cols, rows = 20, 24
    for row in range(rows+1):
        v = row/rows
        for col in range(cols+1):
            u = 2*col/cols-1
            x = u*width*.39*(1-.12*v)
            y = bottom+height*(.49+.51*v)
            z = -depth*.34 + depth*(.09*math.sin(v*math.pi*2)+.09*u*u)
            vertices.append(_blender_point([x,y,z]))
    for row in range(rows):
        for col in range(cols):
            a = row*(cols+1)+col
            faces.append((a,a+1,a+cols+2,a+cols+1))
    mesh = bpy.data.meshes.new("Contoured_Back")
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new("Ergonomic_Lumbar_Back", mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.data.materials.append(mats["fabric"])
    for face in mesh.polygons:
        face.use_smooth = True
    solid = obj.modifiers.new("Back thickness", "SOLIDIFY")
    solid.thickness = min(.025, depth*.05)
    for side in (-1,1):
        add_box(f"Arm_Support_{side}", [width*.045, depth*.06, height*.19],
                [side*width*.44, seat_y+height*.095, 0], mats["metal"], parent=parent)
        add_box(f"Armrest_{side}", [width*.12, depth*.46, height*.035],
                [side*width*.44, seat_y+height*.20, depth*.03], mats["fabric"], parent=parent, bevel=.012)
    for index in range(5):
        angle = index*2*math.pi/5
        radius = min(width,depth)*.43
        arm = add_box(f"Caster_Spoke_{index}", [width*.045, radius, height*.035],
                      [math.sin(angle)*radius/2, bottom+height*.07, math.cos(angle)*radius/2],
                      mats["metal"], parent=parent)
        arm.rotation_euler[2] = angle
        add_cylinder(f"Caster_{index}", min(width,depth)*.055, height*.05,
                     [math.sin(angle)*radius, bottom+height*.025, math.cos(angle)*radius],
                     mats["black"], parent=parent, vertices=24)


def add_quarter_cushion(name,radius,height,position,mat,parent,sign):
    from upholstery_geometry import quarter_sector
    vertices,faces=quarter_sector(radius,height,sign,concave=True)
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata([_blender_point(p) for p in vertices],[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    obj.parent=parent;obj.location=_blender_point(position);obj.data.materials.append(mat)
    for face in obj.data.polygons: face.use_smooth=len(face.vertices)==4
    bevel=obj.modifiers.new('Upholstered edge','BEVEL');bevel.width=min(.025,height*.2);bevel.segments=3
    return obj


def build_sofa(parent, size: Sequence[float], mats: Dict[str, Any], open_arms=(),appearance=None):
    width, depth, height = size
    appearance=appearance or {}
    corner=appearance.get('corner_shape')=='quarter_round' and appearance.get('corner_end') in {'left','right'}
    sign=1 if appearance.get('corner_end')=='right' else -1
    corner_width=min(depth,width*.45) if corner else 0.
    bottom = -height / 2
    seat_h = height * 0.27
    add_box("Sofa_Base", [width, depth, seat_h], [0, bottom + seat_h / 2, 0], mats["fabric"], parent=parent, bevel=0.02, bevel_segments=4)
    if corner:
        # A curved loose cushion still needs a continuous upholstered support
        # deck at its junction. A quarter-shaped plinth left an open black V.
        add_box('Sofa_Corner_Support_Deck',[corner_width,depth,height*.06],
                [sign*(width-corner_width)/2,bottom+seat_h+height*.02,0],mats['fabric'],parent=parent,bevel=.012)
    # Back frame must reach the base; the old 0.58h frame left a 0.15h
    # opening behind the upholstery that became visible while roaming.
    back_h = height * 0.75
    add_box("Sofa_Back_Frame", [width, depth * 0.15, back_h], [0, height / 2 - back_h / 2, -depth * 0.425], mats["fabric"], parent=parent, bevel=0.02, bevel_segments=4)
    arm_width = min(0.18, width * 0.12)
    for side in (-1, 1):
        if side in open_arms: continue
        add_box(f"Sofa_Arm_{side}", [arm_width, depth, height * 0.48], [side * (width / 2 - arm_width / 2), bottom + height * 0.34, 0], mats["fabric"], parent=parent, bevel=0.02, bevel_segments=4)
    # Preserve the observed outer silhouette while giving upholstery an actual
    # seat/back surface. A single monolithic box reads as a wooden rail and
    # loses the photographed cushion divisions from any viewing angle.
    left_inset=0 if -1 in open_arms else arm_width+.02
    right_inset=0 if 1 in open_arms else arm_width+.02
    inner_width = max(.08, width-left_inset-right_inset-corner_width)
    count=appearance.get('seat_count')
    sections=count if type(count) is int and 1<=count<=8 else max(1,min(4,round(inner_width/.72)))
    cushion_width = max(.05, inner_width / sections - .012)
    for index in range(sections):
        x = -width/2+left_inset+(corner_width if corner and sign<0 else 0)+(index+.5)*inner_width/sections
        add_box(f"Sofa_Seat_Cushion_{index}",
                [cushion_width, depth * .74, height * .11],
                [x, bottom + seat_h + height * .045, depth * .045],
                mats["fabric"], parent=parent, bevel=min(.035, height*.04), bevel_segments=4)
        add_box(f"Sofa_Back_Cushion_{index}",
                [cushion_width, depth * .16, height * .44],
                [x, bottom + height * .55, -depth * .32],
                mats["fabric"], parent=parent, bevel=min(.04, height*.045), bevel_segments=4)
    if corner:
        add_quarter_cushion('Sofa_Quarter_Corner_Seat',corner_width*.96,height*.11,
                            [sign*width/2,bottom+seat_h+height*.045,-depth/2],mats['fabric'],parent,sign)


def build_photo_fan(parent,size,mats,appearance,photo_color):
    w,d,h=size;count=appearance.get('blade_count',4)
    if type(count) is not int or not 2<=count<=8: count=4
    finish=material('Photo fan',photo_color,roughness=.65) if photo_color else mats['metal']
    add_cylinder('Fan_Stem',min(w,d)*.025,h*.5,[0,h*.25,0],finish,parent=parent)
    add_cylinder('Fan_Motor',min(w,d)*.11,h*.28,[0,-h*.12,0],finish,parent=parent,vertices=40)
    for i in range(count):
        angle=i*math.tau/count
        obj=add_box(f'Fan_Blade_{i}',[w*.42,d*.10,max(.008,h*.065)],
                    [math.cos(angle)*w*.28,-h*.27,-math.sin(angle)*d*.28],finish,parent=parent,bevel=.005)
        obj.rotation_euler.z=angle


def build_bookshelf(parent, size: Sequence[float], mats: Dict[str, Any]):
    width, depth, height = size
    bottom = -height / 2
    panel = max(0.035, min(0.09, width * 0.055))
    back_depth = max(0.025, min(0.06, depth * 0.10))
    add_box("Bookshelf_Back", [width, back_depth, height], [0, 0, -depth / 2 + back_depth / 2], mats["wood_dark"], parent=parent, bevel=0.008)
    for side in (-1, 1):
        add_box(f"Bookshelf_Side_{side}", [panel, depth, height], [side * (width / 2 - panel / 2), 0, 0], mats["wood"], parent=parent, bevel=0.01)
    for index, ratio in enumerate((0.02, .25, .48, .71, .98)):
        y = bottom + max(panel / 2, min(height - panel / 2, height * ratio))
        add_box(f"Bookshelf_Shelf_{index}", [width, depth, panel], [0, y, 0], mats["wood"], parent=parent, bevel=0.008)


def build_monitor(parent, size, mats):
    w,d,h=size
    add_box('Monitor_Base',[w*.45,d,h*.05],[0,-h*.475,0],mats['metal'],parent=parent,bevel=.008)
    add_box('Monitor_Stand',[w*.08,d*.25,h*.3],[0,-h*.3,-d*.1],mats['metal'],parent=parent,bevel=.008)
    add_box('Monitor_Frame',[w,d*.22,h*.72],[0,h*.14,0],mats['black'],parent=parent,bevel=.012)
    add_box('Monitor_Display',[w*.94,.008,h*.66],[0,h*.14,d*.115],mats['screen'],parent=parent,bevel=.003)

def build_plant(parent,size,mats,foliage_only=False,photo_color=None):
    from foliage_geometry import foliage_geometry
    w,d,h=size
    if not foliage_only:
        add_cone('Plant_Pot',min(w,d)*.28,min(w,d)*.20,h*.3,[0,-h*.35,0],mats['wood_dark'],parent=parent)
    green=material('Visible foliage',photo_color if photo_color is not None else [52,105,53,255],roughness=.8)
    stems,leaves=foliage_geometry((w,d,h if foliage_only else h*.7))
    offset=0 if foliage_only else h*.15
    for start,end in stems:
        bpy.ops.object.select_all(action='DESELECT')
        curve=bpy.data.curves.new('Plant branch','CURVE'); curve.dimensions='3D'
        curve.bevel_depth=min(w,d)*.008; curve.bevel_resolution=2
        spline=curve.splines.new('POLY'); spline.points.add(1)
        for point,position in zip(spline.points,(start,end)):
            point.co=(*_blender_point([position[0],position[1]+offset,position[2]]),1)
        stem=bpy.data.objects.new('Plant_Stem',curve); bpy.context.collection.objects.link(stem)
        stem.parent=parent; stem.data.materials.append(green)
        bpy.context.view_layer.objects.active=stem; stem.select_set(True)
        bpy.ops.object.convert(target='MESH'); stem.select_set(False)
    for leaf in leaves:
        verts=[_blender_point([p[0],p[1]+offset,p[2]]) for p in leaf]
        mesh=bpy.data.meshes.new('Leaf'); mesh.from_pydata(verts,[],[(0,j,j+1) for j in range(1,9)])
        obj=bpy.data.objects.new('Plant_Leaf',mesh); bpy.context.collection.objects.link(obj)
        obj.parent=parent; obj.data.materials.append(green)
        obj.modifiers.new('Leaf thickness','SOLIDIFY').thickness=.003

def build_curtain(parent,size,mats):
    w,d,h=size; vertices=[]; faces=[]; n=64
    for j in range(9):
        for i in range(n+1):
            u=i/n; v=j/8
            # Pleats hang away from the mounting plane below the header;
            # the top edge itself must meet the measured rod/rail plane.
            vertices.append(_blender_point([(u-.5)*w,(v-.5)*h,
                            math.sin(u*math.pi*16)*d*.42*(1-v)**.25]))
    for j in range(8):
        for i in range(n):
            a=j*(n+1)+i; faces.append((a,a+1,a+n+2,a+n+1))
    mesh=bpy.data.meshes.new('Curtain_Folds'); mesh.from_pydata(vertices,[],faces)
    obj=bpy.data.objects.new('Curtain_Draped_Panel',mesh); bpy.context.collection.objects.link(obj)
    obj.parent=parent; obj.data.materials.append(mats['fabric'])
    obj.modifiers.new('Cloth thickness','SOLIDIFY').thickness=.004
    for p in mesh.polygons: p.use_smooth=True

def build_dresser(parent,size,mats):
    w,d,h=size
    add_box('Dresser_Case',size,[0,0,0],mats['wood'],parent=parent,bevel=.012)
    for i in range(4):
        y=-h/2+(i+.5)*h/4
        add_box('Drawer_Front',[w*.96,.025,h*.23],[0,y,d/2],mats['wood_dark'],parent=parent,bevel=.008)
        add_box('Drawer_Handle',[w*.22,.03,.018],[0,y,d/2+.025],mats['metal'],parent=parent,bevel=.004)

def build_book(parent, size: Sequence[float], mats: Dict[str, Any]):
    width, depth, height = size
    if height>width and height>depth:
        cover=min(.012,width*.15)
        add_box('Book_Pages',[max(.006,width-cover*2),depth*.95,height*.96],[0,0,0],mats['fabric_light'],parent=parent,bevel=.002)
        for side in (-1,1):
            add_box('Book_Upright_Cover',[cover,depth,height],[side*(width-cover)/2,0,0],mats['fabric'],parent=parent,bevel=.002)
        add_box('Book_Upright_Spine',[width,.012,height],[0,0,depth/2-.006],mats['fabric'],parent=parent,bevel=.002)
        return
    cover = max(0.008, min(0.018, height * .06))
    add_box("Book_Pages", [width * .94, depth * .94, max(.012, height - cover * 2)], [0, 0, 0], mats["fabric_light"], parent=parent, bevel=0.004)
    add_box("Book_Front_Cover", [width, depth, cover], [0, height / 2 - cover / 2, 0], mats["fabric"], parent=parent, bevel=0.003)
    add_box("Book_Back_Cover", [width, depth, cover], [0, -height / 2 + cover / 2, 0], mats["fabric"], parent=parent, bevel=0.003)
    add_box("Book_Spine", [max(.012, width * .08), depth, height], [-width / 2 + max(.012, width * .08) / 2, 0, 0], mats["fabric"], parent=parent, bevel=0.003)


def build_cabinet(parent, size: Sequence[float], mats: Dict[str, Any], wardrobe: bool = False):
    width, depth, height = size
    bottom = -height / 2
    foot_h = min(.10, height * .08)
    body_h = height - foot_h
    add_box("Cabinet_Body", [width, depth, body_h], [0, bottom + foot_h + body_h / 2, 0], mats["wood"], parent=parent, bevel=min(.025, width * .025))
    door_depth = max(.015, min(.035, depth * .08))
    door_z = depth / 2 + door_depth / 2
    door_gap = min(.018, width * .015)
    for side in (-1, 1):
        door_width = width / 2 - door_gap * 1.5
        add_box(f"Cabinet_Door_{side}", [door_width, door_depth, body_h * .92], [side * (width / 4), bottom + foot_h + body_h / 2, door_z], mats.get("recess",mats["wood_dark"]), parent=parent, bevel=.012)
        add_cylinder(f"Cabinet_Handle_{side}", max(.008, width * .009), min(.16, height * .14), [side * width * .08, bottom + foot_h + body_h * .55, door_z + door_depth], mats["metal"], parent=parent, vertices=24)
    for side in (-1, 1):
        add_box(f"Cabinet_Foot_{side}", [width * .12, depth * .55, foot_h], [side * width * .34, bottom + foot_h / 2, 0], mats["wood_dark"], parent=parent, bevel=.006)
    if not wardrobe:
        add_box("Cabinet_Top", [width, depth, min(.055, height * .07)], [0, height / 2 - min(.055, height * .07) / 2, 0], mats["wood"], parent=parent, bevel=.012)


def build_wall_shelf(parent, size: Sequence[float], mats: Dict[str, Any], variant: str = 'default'):
    width, depth, height = size
    if variant == 'open_shelf' and height >= .15:
        rail = min(.055, width*.09, height*.14)
        add_box('Wall_Shelf_Top',[width,depth,rail],[0,height/2-rail/2,0],mats['wood'],parent=parent,bevel=.006)
        add_box('Wall_Shelf_Bottom',[width,depth,rail],[0,-height/2+rail/2,0],mats['wood'],parent=parent,bevel=.006)
        for side in (-1,1):
            add_box(f'Wall_Shelf_Side_{side}',[rail,depth,height-2*rail],
                    [side*(width/2-rail/2),0,0],mats['wood'],parent=parent,bevel=.006)
        add_box('Wall_Shelf_Recess',[max(.01,width-2*rail),max(.008,depth*.08),
                 max(.01,height-2*rail)],[0,0,-depth*.46],mats['wood_dark'],parent=parent)
        return
    # The detected height describes the slab, not an imaginary bracket assembly.
    # Its top must agree with the support plane used to place books and vessels.
    add_box("Wall_Shelf_Plank", [width, depth, height], [0, 0, 0], mats["wood"],
            parent=parent, bevel=min(.009, height * .12))


def build_mirror(parent, size: Sequence[float], mats: Dict[str, Any]):
    width, depth, height = size
    rail = min(.065, width * .08, height * .08)
    add_box("Mirror_Glass", [width - rail * 2, max(.009, depth * .35), height - rail * 2], [0, 0, depth * .18], mats["glass"], parent=parent, bevel=.008)
    for name, x in (("Left", -width / 2 + rail / 2), ("Right", width / 2 - rail / 2)):
        add_box(f"Mirror_Frame_{name}", [rail, depth, height], [x, 0, 0], mats["wood_dark"], parent=parent, bevel=.008)
    for name, y in (("Top", height / 2 - rail / 2), ("Bottom", -height / 2 + rail / 2)):
        add_box(f"Mirror_Frame_{name}", [width, depth, rail], [0, y, 0], mats["wood_dark"], parent=parent, bevel=.008)


def build_blanket(parent, size, mats, support=None):
    from blanket_geometry import blanket_mesh,cloth_uv
    vertices,faces=blanket_mesh(size,support)
    mesh=bpy.data.meshes.new('Blanket_Cloth_Mesh')
    mesh.from_pydata([_blender_point(v) for v in vertices],[],faces)
    mesh.update()
    obj=bpy.data.objects.new('Blanket_Draped_Cloth',mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent=parent
    obj.data.materials.append(mats['fabric_light'])
    coordinates=cloth_uv(vertices)
    uv=mesh.uv_layers.new(name='DrapedFabricUV')
    for loop in mesh.loops:
        uv.data[loop.index].uv=coordinates[loop.vertex_index%len(coordinates)]
    for polygon in mesh.polygons: polygon.use_smooth=True


def build_window(parent, size: Sequence[float], mats: Dict[str, Any]):
    width, depth, height = size
    rail = min(0.075, width * 0.08, height * 0.08)
    frame_depth = max(0.025, depth)
    glass_depth = min(0.012, frame_depth * 0.28)
    add_box("Window_Glass", [width - rail * 2, glass_depth, height - rail * 2], [0, 0, 0], mats["glass"], parent=parent, bevel=0)
    # Do not invent a centre divider when the detector supplied no pane layout.
    for name, x in (("Left", -width / 2 + rail / 2), ("Right", width / 2 - rail / 2)):
        add_box(f"Window_Frame_{name}", [rail, frame_depth, height], [x, 0, 0], mats["frame"], parent=parent, bevel=0.006)
    for name, y in (("Top", height / 2 - rail / 2), ("Bottom", -height / 2 + rail / 2)):
        add_box(f"Window_Frame_{name}", [width, frame_depth, rail], [0, y, 0], mats["frame"], parent=parent, bevel=0.006)


def build_tv(parent, size: Sequence[float], mats: Dict[str, Any]):
    width, depth, height = size
    add_box("TV_Body", [width, depth, height], [0, 0, 0], mats["black"], parent=parent, bevel=min(0.025, depth * 0.15))
    add_box("TV_Screen", [width * 0.94, min(0.012, depth * 0.12), height * 0.90], [0, 0, depth * 0.49], mats["screen"], parent=parent, bevel=0.006)


def build_profile_lamp(parent,size,profile,profile_name='Lamp'):
    from photo_materials import photo_linear_rgba
    radii=profile['radii']; count=len(radii); segments=48
    vertices=[]; faces=[]
    for j,r in enumerate(radii):
        for i in range(segments):
            angle=i*math.tau/segments
            vertices.append(_blender_point([math.cos(angle)*r*size[0],
                (.5-j/(count-1))*size[2],math.sin(angle)*r*size[1]]))
    for j in range(count-1):
        for i in range(segments):
            a=j*segments+i; b=j*segments+(i+1)%segments
            faces.append((a,b,b+segments,a+segments))
    faces.extend([tuple(reversed(range(segments))),tuple((count-1)*segments+i for i in range(segments))])
    mesh=bpy.data.meshes.new('Observed '+profile_name+' profile'); mesh.from_pydata(vertices,[],faces); mesh.update()
    obj=bpy.data.objects.new('Photo_'+profile_name+'_Profile',mesh); bpy.context.collection.objects.link(obj); obj.parent=parent
    for j in range(count-1):
        mesh.materials.append(material('Photo lamp surface',photo_linear_rgba(profile['colors'][j]),roughness=.75))
    for polygon in mesh.polygons:
        polygon.material_index=min(polygon.index//segments,count-2)
        polygon.use_smooth=polygon.index<(count-1)*segments


def build_lamp(parent, size: Sequence[float], mats: Dict[str, Any], ceiling: bool = False, stem_radius=None, shade_top_ratio=None):
    width, depth, height = size
    radius = max(0.003, min(width, depth) * 0.035)
    if stem_radius is not None and math.isfinite(float(stem_radius)):
        radius=max(.002,min(min(width,depth)*.2,float(stem_radius)))
    if ceiling:
        add_cylinder("Lamp_Cord", radius * 0.32, height * 0.48, [0, height * 0.25, 0], mats["metal"], parent=parent)
        add_cone("Lamp_Shade", min(width, depth) * 0.48, min(width, depth) * 0.18, height * 0.45, [0, -height * 0.22, 0], mats["shade"], parent=parent)
    else:
        bottom = -height / 2
        add_cylinder("Lamp_Base", min(width, depth) * 0.42, height * 0.05, [0, bottom + height * 0.025, 0], mats["metal"], parent=parent)
        add_cylinder("Lamp_Stem", radius, height * 0.68, [0, bottom + height * 0.39, 0], mats["metal"], parent=parent)
        lower_radius=width*.48
        upper_radius=lower_radius*(shade_top_ratio if shade_top_ratio is not None else .58)
        add_cone("Lamp_Shade", lower_radius, upper_radius, height * 0.30, [0, height * 0.34, 0], mats["shade"], parent=parent)


def build_vessel(parent,label,size,photo_color=None):
    """Continuous hollow pottery rather than a bundle of vertical cylinders."""
    width,depth,height=size
    profile=([(-.5,.32),(-.35,.46),(0,.5),(.3,.36),(.5,.23)] if label=='vase'
             else [(-.5,.44),(.18,.44),(.30,.20),(.5,.20)] if label=='bottle'
             else [(-.5,.40),(.5,.46)])
    rings=[(-.5,0.001)]+profile+[(y,max(.001,r-.035)) for y,r in reversed(profile)]+[(-.465,.001)]
    vertices=[]; faces=[]; segments=48
    for y,r in rings:
        for i in range(segments):
            a=i*math.tau/segments
            vertices.append(_blender_point([math.cos(a)*r*width*(.9 if label=='cup' else 1),y*height,math.sin(a)*r*depth]))
    for j in range(len(rings)-1):
        for i in range(segments):
            a=j*segments+i; b=j*segments+(i+1)%segments
            faces.append((a,a+segments,b+segments,b))
    mesh=bpy.data.meshes.new('Vessel_Continuous_Surface'); mesh.from_pydata(vertices,[],faces); mesh.update()
    obj=bpy.data.objects.new('Vessel_Body',mesh); bpy.context.collection.objects.link(obj); obj.parent=parent
    finish=material('Photo ceramic',photo_color if photo_color is not None else [.55,.52,.47,1],roughness=.65)
    mesh.materials.append(finish)
    for face in mesh.polygons: face.use_smooth=True
    if label=='cup':
        for y in (-.22,.22):
            add_box('Cup_Handle',[width*.145,depth*.09,height*.06],[width*.425,y*height,0],finish,parent=parent,bevel=.005)
        add_box('Cup_Handle',[width*.055,depth*.09,height*.48],[width*.47,0,0],finish,parent=parent,bevel=.005)


def build_accessory(parent,label,size,photo_color=None):
    if label in {'vase','cup','bottle'}:
        build_vessel(parent,label,size,photo_color)
        return
    from accessory_assets import parts
    recipe=parts(label)
    lower=[float('inf')]*3; upper=[-float('inf')]*3
    for _,shape,dims,pos,_ in recipe:
        extent=[v/2 for v in dims] if shape=='box' else ([dims[0],dims[1]/2,dims[2]] if shape=='cylinder' else dims)
        for k in range(3): lower[k]=min(lower[k],pos[k]-extent[k]); upper[k]=max(upper[k],pos[k]+extent[k])
    scale=[size[0]/(upper[0]-lower[0]),size[2]/(upper[1]-lower[1]),size[1]/(upper[2]-lower[2])]
    for name,shape,dims,pos,color in recipe:
        p=[(pos[k]-(upper[k]+lower[k])/2)*scale[k] for k in range(3)]
        d=[dims[k]*scale[k] for k in range(3)]
        # Preserve distinct screens, caps and metallic hardware while tinting
        # the visible body. Do not replace every component with one flat color.
        tintable={'Vessel_Wall','Vessel_Base','Bottle_Neck','Cup_Handle','Cushion',
                  'Door_Panel','Blind_Header','Blind_Slat','Roller_Fabric','Frame_Edge',
                  'Keyboard_Base','Mouse_Body','Laptop_Lid','Laptop_Base'}
        if label=='ceiling_fan': tintable.update({'Fan_Stem','Fan_Motor','Fan_Blades_X','Fan_Blades_Z'})
        if photo_color is not None and name in tintable: color=photo_color
        mat=material(name,color,roughness=.65)
        if shape=='box': add_box(name,[d[0],d[2],d[1]],p,mat,parent=parent,bevel=min(.006,min(d)*.08))
        elif shape=='cylinder':
            obj=add_cylinder(name,d[0],d[1],p,mat,parent=parent,vertices=24)
            obj.scale.y=d[2]/max(d[0],.00001)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,radius=1)
            obj=bpy.context.object; obj.name=name; obj.parent=parent
            obj.location=_blender_point(p); obj.scale=(d[0],d[2],d[1]); obj.data.materials.append(mat)
            for face in obj.data.polygons: face.use_smooth=True

def build_object(index: int, spec: Dict[str, Any], objects_root, mats: Dict[str, Any]):
    if spec.get('duplicate_instance_of') is not None:
        return
    from photo_materials import photo_linear_rgba
    label = str(spec.get("label", "object")).lower().replace(" ", "_")
    size = [max(0.005, float(value)) for value in spec.get("size", [0.6, 0.6, 0.6])]
    center = [float(value) for value in spec.get("center", [0, size[2] / 2, -2])]
    parent = add_empty(f"Object_{index:02d}_{label}", center, spec.get("yaw_degrees", 0), objects_root)
    parent["source_label"] = label
    parent["confidence"] = float(spec.get("confidence", 0))
    photo_color=photo_linear_rgba(spec['color']) if spec.get('color_source','').startswith('uploaded_photo') else None
    appearance=spec.get('object_appearance') or {}
    if photo_color is not None:
        mats=dict(mats)
        if label in {'curtain','curtains','rug','book','bed'}:
            mats['fabric']=material('Photo surface',photo_color,roughness=.9,surface='fabric')
        elif label=='pillow':
            mats['fabric_light']=material('Photo pillow',photo_color,roughness=.9,surface='fabric')
        elif label in {'cabinet','bookshelf','wardrobe','dresser','wall_shelf','wall_niche','table','desk','side_table','nightstand','coffee_table','dining_table'}:
            mats['wood']=material('Photo timber',photo_color,roughness=.65,surface='wood')
            # Let geometry and lighting shade the photographed material. A
            # second arbitrary darkening made cabinet fronts nearly black.
            mats['wood_dark']=material('Photo timber trim',photo_color,roughness=.7,surface='wood')
        elif label in {'chair','ergonomic_chair','office_chair','sofa','armchair'}:
            mats['fabric']=material('Photo upholstery',photo_color,roughness=.9,surface='fabric')
    if spec.get("wall"):
        parent["wall"] = str(spec["wall"])
    if label in {'table','coffee_table','dining_table','side_table'} and spec.get('asset_variant')=='round' and photo_color is not None:
        mats=dict(mats)
        mats['fabric_light']=material('Photo round tabletop',photo_color,roughness=.65)
    if label in {'table','coffee_table','dining_table','side_table','desk'} and appearance.get('top_material')=='glass':
        mats=dict(mats)
        glass=material('Observed transparent tabletop',[.85,.90,.92,1],roughness=.12,alpha=.24)
        shader=glass.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Transmission Weight'].default_value=.85;shader.inputs['IOR'].default_value=1.45
        mats['glass']=glass;mats['fabric_light']=glass
    if label=='window' and spec.get('frame_color'):
        mats=dict(mats)
        mats['frame']=material('Photo window frame',photo_linear_rgba(spec['frame_color']),roughness=.55)
    if label=='floor_lamp' and spec.get('lamp_stem_color'):
        mats=dict(mats)
        mats['metal']=material('Photo lamp support',photo_linear_rgba(spec['lamp_stem_color']),roughness=.5,metallic=.15)

    from accessory_assets import ACCESSORIES
    from photo_components import functional_parts,DETAIL_LABELS
    if appearance.get('construction_type')=='display_niche':
        w,d,h=size;thickness=min(w,h)*.09
        for side in (-1,1):
            add_box('Display_Frame_Side_'+str(side),[thickness,d,h],
                [side*(w-thickness)/2,0,0],mats['wood'],parent=parent,bevel=.006)
            add_box('Display_Frame_Ledge_'+str(side),[w,d,thickness],
                [0,side*(h-thickness)/2,0],mats['wood'],parent=parent,bevel=.006)
        add_box('Display_Recess_Back',[w,max(.012,d*.08),h],
            [0,0,-d*.46],mats['wood_dark'],parent=parent,bevel=.003)
    elif label in {'table','coffee_table','dining_table','side_table'} and spec.get('construction_profile')=='open_frame_round':
        build_round_table(parent,size,mats,appearance.get('frame_color_rgb',spec.get('frame_color')))
    elif spec.get('table_profile') and label in {'side_table','nightstand'} and spec.get('construction_support')!='wall':
        build_profile_lamp(parent,size,spec['table_profile'],'Table')
    elif spec.get('lamp_profile') and label in {'table_lamp','lamp'}:
        build_profile_lamp(parent,size,spec['lamp_profile'])
    elif spec.get('photo_components') and label in DETAIL_LABELS-{'sofa','cushion','roller_shade','blinds','curtain','curtains'} and functional_parts(label,spec['photo_components'],spec.get('construction_support','floor'),spec.get('support_style')):
        w,d,h=size
        parts=spec['photo_components']
        if label=='ceiling_lamp' and spec.get('asset_variant') in {'flush_mount','recessed'}:
            from fixture_components import expose_flush_diffuser
            parts=expose_flush_diffuser(parts)
        for i,part in enumerate(parts):
            dims=[part['size'][0]*w,part['size'][1]*d,part['size'][2]*h]
            position=[part['center'][0]*w,part['center'][1]*h,part['center'][2]*d]
            mat=(mats['fabric_light'] if part['kind'] in {'shade','bulb','seat'} else
                 mats['metal'] if part['kind'] in {'handle','stem'} or label in {'ceiling_lamp','wall_lamp','floor_lamp','table_lamp','chandelier'} else mats['wood'])
            if part['kind'] in {'shade','bulb'}:
                mat=fixture_diffuser_material(spec,mat)
            elif part.get('color_rgb'):
                finish='wood' if (spec.get('asset_variant','').startswith('wooden')
                                  and part['kind'] in {'top','shelf'}) else None
                mat=material('Photo component '+part['kind'],photo_linear_rgba(part['color_rgb']),roughness=.7,surface=finish)
            if part['kind']=='top' and appearance.get('top_material')=='glass': mat=mats['glass']
            shape=part.get('shape','box'); name='Photo_Part_'+part['kind']+'_'+str(i)
            if shape=='sphere':
                bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,radius=1)
                obj=bpy.context.object; obj.name=name; obj.parent=parent
                obj.location=_blender_point(position); obj.scale=(dims[0]/2,dims[1]/2,dims[2]/2)
                obj.data.materials.append(mat)
                for face in obj.data.polygons: face.use_smooth=True
            elif shape=='cylinder':
                obj=add_cylinder(name,dims[0]/2,dims[2],position,mat,parent=parent,vertices=32)
                obj.scale.y=dims[1]/max(.001,dims[0])
            else:
                # Upholstered cushions need a rounded volume, not the tiny
                # cabinet-edge bevel used for rigid component panels.
                bevel=min(.08,min(dims)*.35) if part['kind']=='seat' else min(.008,min(dims)*.1)
                obj=add_box(name,dims,position,mat,parent=parent,bevel=bevel)
            obj.rotation_euler.z=math.radians(part.get('yaw_degrees',0))
    elif label=='bench':
        w,d,h=size; seat_h=min(h*.4,.22)
        add_box('Bench_Seat',[w,d,seat_h],[0,h/2-seat_h/2,0],mats['fabric_light'],parent=parent,bevel=min(.08,seat_h*.4))
        _four_legs(parent,w*.88,d*.75,-h/2,h/2-seat_h,mats['wood'],'Bench')
    elif label == 'wall_lamp':
        w,d,h=size
        add_box('Sconce_Backplate',[w*.5,d*.18,h*.8],[0,0,-d*.41],mats['metal'],parent=parent,bevel=.008)
        add_cylinder('Sconce_Diffuser',min(w,d)*.48,h,[0,0,d*.02],fixture_diffuser_material(spec,mats['fabric_light']),parent=parent)
    elif label == 'chandelier':
        build_lamp(parent,size,mats,ceiling=True)
    elif label=='cushion' and spec.get('support_label') in {'sofa','armchair'}:
        cushion_mats=dict(mats)
        if photo_color is not None: cushion_mats['fabric_light']=material('Photo loose cushion',photo_color,roughness=.9,surface='fabric')
        if appearance.get('cushion_shape')=='bolster':
            w,d,h=size
            obj=add_cylinder('Bolster_Cushion',h/2,w,[0,0,0],cushion_mats['fabric_light'],parent=parent,vertices=48)
            obj.scale.y=d/max(.005,h);obj.rotation_euler.y=math.pi/2
        else: build_pillow(parent,size,cushion_mats,spec.get('pillow_pose','upright'))
    elif label=='ceiling_fan':
        build_photo_fan(parent,size,mats,appearance,photo_color)
    elif label in ACCESSORIES:
        build_accessory(parent,label,size,photo_color)
    elif label == "bed":
        bed_mats=dict(mats)
        if spec.get('asset_variant')=='upholstered': bed_mats['wood']=mats['fabric']
        elif spec.get('asset_variant')=='metal': bed_mats['wood']=mats['metal']
        elif photo_color is not None:
            bed_mats['wood']=material('Photo bed frame',photo_color,roughness=.7,
                                      surface=None if spec.get('asset_variant')=='painted_slatted' else 'wood')
        build_bed(parent, size, bed_mats,spec.get('asset_variant','default'),spec.get('headboard_mode')!='architectural')
    elif label == "pillow":
        mats=dict(mats)
        mats['fabric_light']=fabric_pattern_material(spec,mats['fabric_light'])
        build_pillow(parent, size, mats,spec.get('pillow_pose'))
    elif label in {"table", "desk", "dining_table", "coffee_table"}:
        if spec.get('asset_variant')=='round':
            build_round_table(parent,size,mats,appearance.get('frame_color_rgb',spec.get('frame_color')))
        else:
            build_table(parent, size, mats)
    elif label in {"side_table", "nightstand"}:
        # Valid components were handled above; never bypass the support check
        # here for an old cached, incomplete part description.
        build_table(parent, size, mats, side_table=True,variant=spec.get('asset_variant','default'))
    elif label in {"ergonomic_chair", "office_chair"}:
        build_ergonomic_chair(parent, size, mats)
    elif label == 'armchair':
        build_sofa(parent,size,mats)
    elif label in {"chair", "stool"}:
        build_chair(parent, size, mats)
    elif label == "sofa":
        build_sofa(parent, size, mats,spec.get('open_arms',()),appearance)
    elif label == "window":
        opaque=dict(mats)
        opaque['glass']=material('Opaque window glass', [118,169,185,255],roughness=.4,alpha=1)
        build_window(parent, size, opaque)
    elif label in {"tv", "wall_tv"}:
        build_tv(parent, size, mats)
    elif label in {"floor_lamp", "table_lamp", "lamp"}:
        if spec.get('lamp_profile'): build_profile_lamp(parent,size,spec['lamp_profile'])
        elif label=='table_lamp':
            # A tabletop fixture is not a scaled floor lamp with a tall pole.
            # Keep housing, short neck and broad diffuser independently editable.
            w,d,h=size
            add_cylinder('Table_Lamp_Housing',w*.24,h*.46,[0,-h*.27,0],mats['wood'],parent=parent,vertices=40)
            add_cylinder('Table_Lamp_Neck',w*.045,h*.18,[0,h*.03,0],mats['metal'],parent=parent)
            add_cylinder('Table_Lamp_Drum_Shade',w*.5,h*.36,[0,h*.32,0],fixture_diffuser_material(spec,mats['shade']),parent=parent,vertices=48)
        else: build_lamp(parent, size, mats,stem_radius=spec.get('lamp_stem_radius'),shade_top_ratio=spec.get('lamp_shade_top_ratio'))
    elif label == "ceiling_lamp":
        if spec.get('asset_variant') in {'flush_mount','recessed'}:
            add_cylinder('Ceiling_Fixture_Rim',min(size[:2])*.5,size[2],[0,0,0],mats['metal'],parent=parent,vertices=48)
            add_cylinder('Ceiling_Light_Diffuser',min(size[:2])*.44,.008,[0,-size[2]/2,0],fixture_diffuser_material(spec,mats['shade']),parent=parent,vertices=48)
        else:
            build_lamp(parent, size, mats, ceiling=True)
    elif label in {"bookshelf", "bookcase"}:
        build_bookshelf(parent, size, mats)
    elif label in {"book", "books", "magazine"}:
        build_book(parent, size, mats)
    elif label == 'monitor':
        build_monitor(parent,size,mats)
    elif label == 'plant':
        build_plant(parent,size,mats,spec.get('foliage_only',False),photo_color)
    elif label == 'dresser':
        build_dresser(parent,size,mats)
    elif label in {"cabinet"}:
        cabinet_mats=dict(mats)
        if (spec.get('asset_variant')=='slatted' and
                spec.get('recess_color_source')=='uploaded_photo_slat_recess'):
            cabinet_mats['recess']=material('Photo timber recess',
                photo_linear_rgba(spec['recess_color']),roughness=.95)
        build_cabinet(parent, size, cabinet_mats)
        if spec.get('asset_variant')=='slatted':
            w,d,h=size; count=int(spec.get('slat_count',24))
            for i in range(count):
                add_box('Cabinet_Vertical_Slat',[w/count*.48,.035,h*.9],
                        [-w/2+(i+.5)*w/count,h*.02,d/2+.045],mats['wood'],parent=parent,bevel=.003)
    elif label == "wardrobe":
        build_cabinet(parent, size, mats, wardrobe=True)
    elif label == "wall_shelf":
        build_wall_shelf(parent, size, mats, spec.get('asset_variant','default'))
    elif label == "wall_niche":
        w,d,h=size; frame=min(.055,w*.09,h*.1)
        add_box('Niche_Back',[w,d*.14,h],[0,0,-d*.43],mats['wood_dark'],parent=parent,bevel=.003)
        for side in (-1,1):
            add_box(f'Niche_Side_{side}',[frame,d,h],[side*(w-frame)/2,0,0],mats['wood'],parent=parent,bevel=.003)
        for side in (-1,1):
            add_box(f'Niche_Rail_{side}',[w-2*frame,d,frame],[0,side*(h-frame)/2,0],mats['wood'],parent=parent,bevel=.003)
    elif label == "wall_panel":
        finish=material('Photo architectural wall panel',photo_color,roughness=.68,surface='wood') if photo_color else mats['wood']
        add_box('Wall_Panel',size,[0,0,0],finish,parent=parent,bevel=min(.004,min(size)*.1))
    elif label == "air_conditioner":
        w,d,h=size
        body=material('Photo AC housing',photo_color,roughness=.58) if photo_color else mats['neutral']
        add_box('AC_Housing',[w,d,h*.82],[0,h*.09,0],body,parent=parent,bevel=min(.04,h*.15))
        add_box('AC_Dark_Vent',[w*.86,.012,h*.13],[0,-h*.33,d*.5+.006],mats['black'],parent=parent,bevel=.003)
    elif label in {"mirror", "wall_mirror"}:
        build_mirror(parent, size, mats)
    elif label in {"blanket", "duvet", "throw"}:
        cloth_mats=dict(mats)
        if spec.get('color_source') == 'uploaded_photo_cloth_region':
            cloth_mats['fabric_light']=material('Photo cloth',photo_color,roughness=.92,surface='fabric')
        cloth_mats['fabric_light']=fabric_pattern_material(spec,cloth_mats['fabric_light'])
        build_blanket(parent,size,cloth_mats,spec.get('cloth_support'))
    elif label in {'curtain','curtains'}:
        build_curtain(parent,size,mats)
    elif label in {"rug"}:
        add_box(f"{label}_Fabric", size, [0, 0, 0], mats["fabric"], parent=parent, bevel=min(0.018, min(size) * 0.15))
    elif label in {"picture", "wall_art"}:
        finish=material('Photo artwork',photo_color,roughness=.85) if photo_color else mats['wood_dark']
        add_box("Picture_Frame", size, [0, 0, 0], finish, parent=parent, bevel=0.01)
    else:
        # Unknown objects stay honest and volumetric, with a restrained bevel
        # instead of the former inflated sphere-derived proxy.
        add_box("Visible_Object_Volume", size, [0, 0, 0], mats["neutral"], parent=parent, bevel=min(0.02, min(size) * 0.06))


def build_scene(payload: Dict[str, Any], output_path: Path, preview_path: Path | None = None) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            datablocks.remove(datablock)
    MATERIALS.clear()
    TEXTURES.clear()

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "BLENDER_EEVEE"

    room = [float(value) for value in payload["room_dimensions"]]
    width, length, height = room
    colors = payload.get("surface_colors", {})
    floor_evidence=payload.get('floor_material',{})
    floor_kind=floor_evidence.get('kind','neutral')
    if floor_kind not in {'carpet','tile','wood','neutral'}: floor_kind='neutral'
    from photo_materials import photo_linear_rgba
    floor_rgb=floor_evidence.get('color',colors.get('floor'))
    mats = {
        "floor": material("Photo floor" if floor_rgb else "Floor", photo_linear_rgba(floor_rgb) if floor_rgb else _rgba(None,[142,110,78,255]), roughness=.96 if floor_kind=='carpet' else .62, surface=floor_kind),
        "wall": material("Photo wall", photo_linear_rgba(colors.get("wall") or [224,220,212]), roughness=0.78, surface="wall"),
        "ceiling": material("Photo ceiling", photo_linear_rgba(colors.get("ceiling") or [238,237,232]), roughness=0.82),
        "wood": material("Natural wood", [142, 86, 48, 255], roughness=0.52, surface="wood"),
        "wood_dark": material("Dark wood", [76, 48, 33, 255], roughness=0.58, surface="wood"),
        "door_wood": material("Entry door wood", [116, 70, 39, 255], roughness=0.50, surface="wood"),
        "fabric": material("Upholstery", [115, 128, 137, 255], roughness=0.88, surface="fabric"),
        "fabric_light": material("Light fabric", [225, 221, 211, 255], roughness=0.92, surface="fabric"),
        "fabric_shadow": material("Fabric seam", [193, 189, 181, 255], roughness=0.94, surface="fabric"),
        "metal": material("Brushed metal", [77, 78, 80, 255], roughness=0.32, metallic=0.72),
        "frame": material("Window frame", [224, 224, 219, 255], roughness=0.44, metallic=0.08),
        "glass": material("Window glass", [118, 169, 185, 255], roughness=0.16, alpha=0.32),
        "black": material("TV body", [18, 20, 23, 255], roughness=0.30),
        "screen": material("TV screen", [27, 43, 54, 255], roughness=0.18, metallic=0.12),
        "shade": material("Lamp shade", [229, 204, 157, 255], roughness=0.82),
        "neutral": material("Detected neutral object", [158, 165, 171, 255], roughness=0.68, surface="neutral"),
    }

    room_root = add_empty("Room", [0, 0, 0])
    architecture = add_empty("Room_Envelope", [0, 0, 0], parent=room_root)
    objects_root = add_empty("Objects", [0, 0, 0], parent=room_root)
    wall_thickness = 0.08
    floor_center_z = -length / 2 + 0.15
    floor_mesh=add_box("Floor", [width, length, 0.05], [0, -0.025, floor_center_z], mats["floor"], parent=architecture)
    floor_mesh['surface_kind']=floor_kind
    if floor_kind in {'carpet','tile'}:
        tile_metres=.6 if floor_kind=='tile' else .45
        uv=floor_mesh.data.uv_layers.active
        for loop in floor_mesh.data.loops:
            p=floor_mesh.data.vertices[loop.vertex_index].co
            uv.data[loop.index].uv=(p.x/tile_metres,p.y/tile_metres)
    from wall_openings import wall_panels
    for wall in ('back','left','right'):
        openings=[]
        for spec in payload.get('objects',[]):
            if spec.get('label')!='window' or (spec.get('wall') or spec.get('wall_contact'))!=wall: continue
            pos=spec['center']; size=spec['size']
            along=pos[0] if wall=='back' else pos[2]
            # Leave a small overlap behind the frame to avoid cracks.
            half=max(.01,size[0]/2-min(.04,size[0]*.04))
            openings.append((along-half,along+half,pos[1]-size[2]/2+.025,pos[1]+size[2]/2-.025))
        start,end=(-width/2,width/2) if wall=='back' else (-length+.15,.15)
        for index,(a,b,lo,hi) in enumerate(wall_panels(start,end,height,openings)):
            dims=[b-a,wall_thickness,hi-lo] if wall=='back' else [wall_thickness,b-a,hi-lo]
            pos=[(a+b)/2,(lo+hi)/2,-length+.15] if wall=='back' else [(-1 if wall=='left' else 1)*width/2,(lo+hi)/2,(a+b)/2]
            name=f'{wall.title()}_Wall'+(f'_Panel_{index}' if openings else '')
            add_box(name,dims,pos,mats['wall'],parent=architecture)
    build_wall_finishes(payload.get('wall_surfaces',[]),room,architecture,payload.get('objects',[]))
    # Close the camera-side opening with the same wall finish as the rest of
    # the room. The door is a distinct wood panel, flush with its inner face.
    front_wall_z = 0.15
    add_box("Front_Wall", [width, wall_thickness, height], [0, height / 2, front_wall_z], mats["wall"], parent=architecture)
    door_width = min(0.95, max(0.72, width * 0.22))
    door_height = min(2.15, max(1.90, height * 0.78))
    door_depth = 0.035
    door_z = front_wall_z - wall_thickness / 2 - door_depth / 2 - 0.002
    add_box(
        "Entry_Door",
        [door_width, door_depth, door_height],
        [0, door_height / 2, door_z],
        mats["door_wood"],
        parent=architecture,
        bevel=0.012,
        bevel_segments=4,
    )
    frame_width = 0.055
    frame_depth = door_depth + 0.012
    frame_z = door_z - 0.004
    for side in (-1, 1):
        add_box(
            f"Entry_Door_Frame_{'Left' if side < 0 else 'Right'}",
            [frame_width, frame_depth, door_height + frame_width],
            [side * (door_width / 2 + frame_width / 2), door_height / 2, frame_z],
            mats["wood_dark"],
            parent=architecture,
            bevel=0.006,
        )
    add_box(
        "Entry_Door_Frame_Top",
        [door_width + frame_width * 2, frame_depth, frame_width],
        [0, door_height + frame_width / 2, frame_z],
        mats["wood_dark"],
        parent=architecture,
        bevel=0.006,
    )
    add_cylinder(
        "Entry_Door_Handle",
        0.025,
        0.09,
        [door_width * 0.34, door_height * 0.50, door_z - door_depth / 2 - 0.025],
        mats["metal"],
        parent=architecture,
    )
    add_box("Ceiling", [width, length, 0.05], [0, height, floor_center_z], mats["ceiling"], parent=architecture)

    for index, spec in enumerate(payload.get("objects", [])):
        build_object(index, spec, objects_root, mats)

    for obj in bpy.context.scene.objects:
        if obj.type=='MESH': ensure_texture_uvs(obj.data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        export_yup=True,
        export_apply=True,
        export_normals=True,
        export_materials="EXPORT",
        export_extras=True,
    )
    if not output_path.exists() or output_path.stat().st_size < 512:
        raise RuntimeError("Blender did not produce a valid GLB file.")

    if preview_path is not None:
        # Optional local QA render. The normal web-app export never enters
        # this branch; previews live outside spaces and therefore not Recent.
        from mathutils import Vector, Matrix
        room_length = float(room[1])
        camera_data = bpy.data.cameras.new('QA interior camera')
        camera_obj = bpy.data.objects.new('QA interior camera', camera_data)
        bpy.context.collection.objects.link(camera_obj)
        camera_obj.location = (0, .18, min(1.55, float(room[2]) * .55))
        aim = Vector((0, room_length * .62, min(1.45, float(room[2]) * .5)))
        camera_obj.rotation_euler = (aim - camera_obj.location).to_track_quat('-Z', 'Y').to_euler()
        camera_data.type = 'PERSP'
        camera_data.lens = 20
        bpy.context.scene.camera = camera_obj
        light_data = bpy.data.lights.new('QA soft fill', type='AREA')
        light_obj = bpy.data.objects.new('QA soft fill', light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = (0, room_length * .4, max(2, float(room[2]) - .2))
        light_data.energy = 450
        light_data.shape = 'DISK'
        light_data.size = max(2, float(room[0]) * .6)
        bpy.context.scene.world.color = (.35, .35, .35)
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_x = 1200
        bpy.context.scene.render.resolution_y = 675
        # Use the same calibrated pose and intrinsics as the web viewer when
        # present. This is a geometry comparison, not a claim of relighting parity.
        calibration = payload.get('preview_camera') or {}
        if calibration.get('position') and calibration.get('intrinsics'):
            position = calibration['position']
            yaw = math.radians(float(calibration.get('yaw_degrees', 0)))
            camera_obj.location = (position[0], -position[2], position[1])
            direction = Vector((-math.sin(yaw), math.cos(yaw), 0))
            camera_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            if calibration.get('rotation_matrix'):
                # Map Y-up camera orientation into Blender's Z-up world.
                conversion=Matrix(((1,0,0),(0,0,-1),(0,1,0)))
                camera_obj.rotation_euler=(conversion@Matrix(calibration['rotation_matrix'])).to_euler()
            intrinsics = calibration['intrinsics']
            shape = calibration.get('source_shape', [900,1200])
            bpy.context.scene.render.resolution_x = 1200
            bpy.context.scene.render.resolution_y = round(1200 * shape[0] / shape[1])
            camera_data.sensor_fit = 'HORIZONTAL'
            camera_data.lens = float(intrinsics[0][0]) * camera_data.sensor_width
            camera_data.shift_x = .5 - float(intrinsics[0][2])
            camera_data.shift_y = (float(intrinsics[1][2]) - .5) * shape[0] / shape[1]
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(preview_path)
        bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", help="Optional local QA PNG; never added to Recent")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    build_scene(payload, Path(args.output), Path(args.preview) if args.preview else None)
    print(json.dumps({"status": "ok", "output": str(Path(args.output)), "objects": len(payload.get("objects", []))}))


if __name__ == "__main__":
    main()
