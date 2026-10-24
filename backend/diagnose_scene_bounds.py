"""Print only geometry rejected by primary analysis; never creates a Recent entry."""
import json
import sys
import cv2
import reconstruction.groq_scene_engine as groq

original=groq.parse_scene
def traced(data,shape):
    try:
        return original(data,shape)
    except ValueError:
        print(json.dumps({'room':data.get('room'),'objects':data.get('objects')},indent=2),flush=True)
        raise

if __name__=='__main__':
    groq.parse_scene=traced
    result=groq.GroqSceneEngine().analyze_scene(cv2.imread(sys.argv[1]))
    print(json.dumps({'success':True,'objects':len(result.objects),'exported':False}))
