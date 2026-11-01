import React, { useState, useEffect, useRef } from 'react';
import { ReconstructionEngine } from '../types';
import { checkRuntimeWithDeadline } from '../services/runtimeCheck';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (files: File[], name: string, engine: string, options: {knownRoomWidthM?: number; photographic?: boolean}) => void;
  engines: ReconstructionEngine[];
  isSubmitting: boolean;
  onRefreshEngines?: (engineId?: string) => Promise<void>;
}

export const UploadModal: React.FC<UploadModalProps> = ({isOpen,onClose,onSubmit,engines,isSubmitting,onRefreshEngines}) => {
  const [files,setFiles]=useState<File[]>([]);
  const [name,setName]=useState('');
  const [error,setError]=useState('');
  const [checking,setChecking]=useState(false);
  const dialog=useRef<HTMLDivElement>(null);
  const generation=useRef(0);
  const engine=engines.find(item=>item.id==='resplat_experimental');
  useEffect(()=>{
    generation.current++;
    if(!isOpen){setFiles([]);setName('');setError('');setChecking(false);return;}
    const previous=document.activeElement as HTMLElement;
    const frame=requestAnimationFrame(()=>dialog.current?.querySelector<HTMLInputElement>('input')?.focus());
    return ()=>{cancelAnimationFrame(frame);previous?.focus();};
  },[isOpen]);
  if(!isOpen)return null;
  const recheck=async()=>{
    if(checking||!onRefreshEngines)return;
    const current=generation.current;setChecking(true);setError('');
    try{await checkRuntimeWithDeadline(()=>onRefreshEngines('resplat_experimental'));}
    catch(e){if(current===generation.current)setError(e instanceof Error?e.message:'Runtime check failed');}
    finally{if(current===generation.current)setChecking(false);}
  };
  const submit=(event:React.FormEvent)=>{
    event.preventDefault();
    const videos=files.filter(file=>/\.(mov|mp4|avi|mkv|webm)$/i.test(file.name));
    if(!(videos.length===1&&files.length===1)&&!(videos.length===0&&files.length>=5&&files.length<=10)){
      setError('Choose 5–10 overlapping photos or one video, without mixing them.');return;
    }
    if(files.reduce((sum,file)=>sum+file.size,0)>500*1024*1024){setError('The combined upload limit is 500 MB.');return;}
    if(!engine?.can_run){setError('Configure and check the CUDA runtime first.');return;}
    onSubmit(files,name.trim()||'New room','resplat_experimental',{});
  };
  return <div className="fixed inset-0 z-[100] bg-black/75 flex items-center justify-center p-4">
    <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby="capture-title"
      className="w-full max-w-2xl rounded-3xl border border-white/15 bg-[#1c1916] p-6 text-white shadow-2xl"
      onKeyDown={event=>{
        if(event.key==='Escape'&&!isSubmitting)onClose();
        if(event.key==='Tab'){
          const items=Array.from(dialog.current?.querySelectorAll<HTMLElement>('input:not([disabled]),button:not([disabled])')||[]);
          const first=items[0],last=items[items.length-1];
          if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
          else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
        }
      }}>
      <div className="flex items-center justify-between mb-6"><h2 id="capture-title" className="text-xl font-bold">Create 3D Space</h2>
        <button type="button" aria-label="Close" disabled={isSubmitting} onClick={onClose}>×</button></div>
      <form onSubmit={submit} className="space-y-5">
        <label className="block">Room name<input value={name} onChange={e=>setName(e.target.value)} maxLength={100}
          className="block w-full mt-2 p-3 rounded-xl bg-black/30 border border-white/15" placeholder="My room" /></label>
        <div className="p-4 border border-orange-600/60 rounded-xl"><strong>COLMAP + ReSplat</strong>
          <p className="mt-2 text-sm text-gray-300">Choose 5–10 overlapping photos or one 1–90 second single-room video. Portrait and landscape are supported. Keep the lens and zoom fixed.</p>
          <input className="block w-full mt-4" aria-label="Room photos or video" type="file" multiple
            accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.mp4,.mov,.avi,.mkv,.webm"
            disabled={isSubmitting} onChange={e=>{setFiles(Array.from(e.target.files||[]));setError('');}} />
          <p className="mt-2 text-sm">{files.length} file(s) selected · 500 MB maximum combined</p>
        </div>
        <div role="status" className="text-sm text-amber-200">
          <p>{engine?.availability_message||'Loading runtime status…'}</p>
          <p className="mt-2">CUDA is required. This app does not start a GPU VM automatically. Results require review; scale and unseen surfaces are unverified.</p>
          {onRefreshEngines&&<button type="button" onClick={()=>void recheck()} disabled={checking||isSubmitting} className="underline mt-2">
            {checking?'Checking runtime… (up to 50 seconds)':'Reconnect running GPU'}</button>}
        </div>
        {error&&<p role="alert" className="text-red-300">{error}</p>}
        <div className="flex justify-end gap-4"><button type="button" disabled={isSubmitting} onClick={onClose}>Cancel</button>
          <button className="px-5 py-3 rounded-xl bg-[#d5360c] disabled:opacity-40" disabled={isSubmitting||checking||!files.length||!engine?.can_run}>
            {isSubmitting?'Uploading…':'Start 3D Reconstruction'}</button></div>
      </form>
    </div>
  </div>;
};
