"""Stage an unpublished calibrated OpenMVS comparison, without default adoption."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import struct
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job');parser.add_argument('benchmark',type=Path)
    parser.add_argument('--asset-stem',default='scene_textured')
    args=parser.parse_args()
    if not all(c.isalnum() or c in '-_' for c in args.asset_stem):raise ValueError('Invalid asset stem')
    if not all(c.isalnum() or c in '-_' for c in args.job):raise ValueError('Invalid job')
    room=ROOT/'backend/storage/spaces'/args.job
    metadata=json.loads((room/'metadata.json').read_text(encoding='utf-8'))
    previous=room/'gaussian'/metadata['gaussian']['ply_url'].split('/')[-2]
    config=json.loads((previous/'config.json').read_text(encoding='utf-8'))
    report=json.loads((args.benchmark/'report.json').read_text(encoding='utf-8'))
    if report['status']!='completed_standalone_candidate':raise ValueError('Incomplete OpenMVS result')
    data=(args.benchmark/(args.asset_stem+'.glb')).read_bytes()
    magic,version,length=struct.unpack('<III',data[:12]);size,kind=struct.unpack('<II',data[12:20])
    if (magic,version,length)!=(0x46546c67,2,len(data)):raise ValueError('Invalid GLB')
    document=json.loads(data[20:20+size]);binary=bytearray(data[28+size:])
    if len(document['buffers'])!=1:raise ValueError('Unsupported multi-buffer GLB')
    # Native OpenMVS GLB references a PNG. Embed only a same-directory image;
    # never expose arbitrary filesystem references through the viewer.
    for image in document['images']:
        name=image.pop('uri');texture=(args.benchmark/name).resolve()
        if texture.parent!=args.benchmark.resolve() or texture.suffix.lower()!='.png':raise ValueError('Unsafe texture reference')
        image.pop('name',None)
        while len(binary)%4:binary.append(0)
        content=texture.read_bytes();view=len(document['bufferViews'])
        document['bufferViews'].append(dict(buffer=0,byteOffset=len(binary),byteLength=len(content)));binary.extend(content)
        image.update(bufferView=view,mimeType='image/png')
    while len(binary)%4:binary.append(0)
    document['buffers'][0]['byteLength']=len(binary)
    transform=np.asarray(config['model_transform'],float)
    scene=document['scenes'][document.get('scene',0)];nodes=scene['nodes']
    document['nodes'].append(dict(name='Recovered_COLMAP_frame',children=nodes,matrix=transform.T.ravel().tolist()))
    scene['nodes']=[len(document['nodes'])-1]
    for material in document.get('materials',[]):
        material.setdefault('extensions',{})['KHR_materials_unlit']={};material['alphaMode']='OPAQUE'
    document.setdefault('extensionsUsed',[])
    if 'KHR_materials_unlit' not in document['extensionsUsed']:document['extensionsUsed'].append('KHR_materials_unlit')
    triangles=sum(document['accessors'][primitive['indices']]['count']//3 for mesh in document['meshes'] for primitive in mesh['primitives'])
    revision=uuid.uuid4().hex;target=room/'gaussian'/revision;target.mkdir()
    for name in ['scene.ply','scene.ply.gz']:
        if (previous/name).exists():shutil.copyfile(previous/name,target/name)
    header=json.dumps(document,separators=(',',':')).encode();header+=b' '*((-len(header))%4)
    (target/'surface-99.glb').write_bytes(struct.pack('<III',0x46546c67,2,28+len(header)+len(binary))+struct.pack('<II',len(header),0x4e4f534a)+header+struct.pack('<II',len(binary),0x004e4942)+binary)
    config=copy.deepcopy(config)
    config['fused_surface']=dict(asset='surface-99.glb',method='OpenMVS standalone evaluation',triangles=triangles,views=len(config['cameras']),provider_calls=0,license='AGPL-3.0')
    config['notice']='Unpublished OpenMVS comparison. Existing collision/enclosure held constant to isolate geometry; primary pipeline is not changed.'
    (target/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (target/'comparison.json').write_text(json.dumps(dict(benchmark=str(args.benchmark),asset_stem=args.asset_stem,published=False,baseline=str(previous),triangles=triangles),indent=2),encoding='utf-8')
    print(revision,flush=True)


if __name__=='__main__':main()
