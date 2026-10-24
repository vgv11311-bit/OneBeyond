import React, { useEffect, useRef, useState } from 'react';
import { SpaceJob } from '../../types';
import { ArrowLeft, Layers } from 'lucide-react';

export const GaussianViewer: React.FC<{job: SpaceJob; onExit: () => void; onFallback: (reason: string) => void}> = ({job,onExit,onFallback}) => {
  const frame = useRef<HTMLIFrameElement>(null);
  const [ready,setReady] = useState(false);
  const url = job.gaussian?.viewer_url;
  const validUrl = !!url && /^\/api\/spaces\/[A-Za-z0-9_-]+\/gaussian\/[a-f0-9]{32}\/viewer$/.test(url);
  const validExport = !!job.gaussian?.ply_url && validUrl && job.gaussian.ply_url === url!.replace(/\/viewer$/, '/scene.ply');
  const meshExport = !!job.gaussian?.mesh_url && validUrl && job.gaussian.mesh_url === url!.replace(/\/viewer$/, '/surface-99.glb');
  useEffect(() => {
    if (!validUrl) {onFallback('Invalid photographic result URL.');return;}
    const timer = window.setTimeout(() => onFallback('Photographic viewer did not load in time.'), 90000);
    const receive = (event: MessageEvent) => {
      if (event.origin !== location.origin || event.source !== frame.current?.contentWindow) return;
      if (event.data?.type === 'one-beyond-splat-ready') {clearTimeout(timer);setReady(true);}
      if (event.data?.type === 'one-beyond-splat-error') {clearTimeout(timer);onFallback('Photographic rendering is unavailable on this device.');}
    };
    window.addEventListener('message', receive);
    return () => {clearTimeout(timer);window.removeEventListener('message', receive);};
  }, [job.job_id,onFallback,validUrl]);
  // Only accept this application's artifact route, never external/embed URLs.
  if (!validUrl) return null;
  return <div className="fixed inset-0 z-50 overflow-hidden bg-slate-950 text-white">
    <iframe ref={frame} title="Photographic room reconstruction" src={url} allow="fullscreen" className="absolute inset-0 block h-full w-full border-0" onError={() => onFallback('Photographic viewer could not be loaded.')} />
    {!ready && <div role="status" className="absolute bottom-5 left-5 rounded bg-black/80 p-3">Loading photographic reconstruction…</div>}
    {job.gaussian?.requires_review && <div role="status" className="absolute bottom-20 left-4 max-w-sm rounded bg-black/85 p-3 text-xs text-amber-200">Experimental {job.engine_id==='resplat_experimental'?'ReSplat':'VGGT'} preview — quality unverified. {job.gaussian.saved_to_recents ? 'Saved in Recents for review, not quality approved.' : 'Not saved in Recents.'} Inspect translated viewpoints; geometry and scale are unverified.</div>}
    <div className="absolute right-4 top-4 flex gap-2">
      {validExport && <a href={job.gaussian!.ply_url} download={`${job.job_id}.ply`} className="glass-panel rounded-full border border-white/10 px-3 py-2.5 text-xs font-medium text-gray-300 hover:bg-white/10 hover:text-white">Download splat</a>}
      {meshExport && <a href={job.gaussian!.mesh_url} download={`${job.job_id}-experimental.glb`} className="glass-panel rounded-full border border-white/10 px-3 py-2.5 text-xs font-medium text-gray-300">Download preview mesh</a>}
      {job.glb_url && <button className="glass-panel flex items-center gap-2 rounded-full border border-white/10 px-3 py-2.5 text-xs font-medium text-gray-300 hover:bg-white/10 hover:text-white active:scale-[.97]" onClick={() => onFallback('')}><Layers className="h-4 w-4"/>Editable room</button>}
      <button className="glass-panel flex items-center gap-2 rounded-full border border-white/10 px-3 py-2.5 text-xs font-medium text-gray-300 hover:bg-white/10 hover:text-white active:scale-[.97]" onClick={onExit}><ArrowLeft className="h-4 w-4"/>Exit room</button>
    </div>
  </div>;
};
