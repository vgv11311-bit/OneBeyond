"""Focused QA of saved position evidence; never exports or adds Recent entries."""
import argparse
import base64
import json
import time
from pathlib import Path
import cv2
from reconstruction.groq_scene_engine import GroqSceneEngine
from reconstruction import provider_resilience

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('position_cache')
    parser.add_argument('--gemini-only',action='store_true')
    args=parser.parse_args()
    image=cv2.imread(args.image)
    data=json.loads(Path(args.position_cache).read_text())['value']['result']
    engine=GroqSceneEngine()
    engine._analysis_deadline=time.monotonic()+180
    engine.analysis_status=lambda message:print(message,flush=True)
    if args.gemini_only:
        import os
        from reconstruction.gemini_scene_engine import GeminiSceneEngine
        from provider_credentials import gemini_backup_key
        engine._groq_daily_unavailable=True
        engine._detail_checker=GeminiSceneEngine()
        engine._detail_checker_key=os.environ.get('GEMINI_API_KEY','').strip() or gemini_backup_key()
        if not engine._detail_checker_key: raise RuntimeError('Gemini is not configured')
    original=engine._request_json
    def request(*values,**kwargs):
        result=original(*values,**kwargs)
        print('BED RESPONSE '+json.dumps(result),flush=True)
        return result
    engine._request_json=request
    encoded=provider_resilience.encode_overview(image)
    data,position_status=engine.verify_spatial_layout(data,base64.b64encode(encoded).decode(),image.shape,'groq','')
    print('POSITION STATUS '+position_status,flush=True)
    _,status=engine.verify_bed_surfaces(data,base64.b64encode(encoded).decode(),image.shape,'groq','')
    print('STATUS '+status,flush=True)

if __name__=='__main__':main()
