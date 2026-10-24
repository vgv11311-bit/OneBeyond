import React, { useEffect, useState, useRef } from 'react';
import { X, Upload, Image as ImageIcon, Cpu, ArrowRight, AlertCircle, FileCheck, Layers } from 'lucide-react';
import { ReconstructionEngine } from '../types';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (files: File[], spaceName: string, engineId: string, options: { knownRoomWidthM?: number; photographic?: boolean }) => void;
  engines: ReconstructionEngine[];
  isSubmitting: boolean;
  onRefreshEngines?: () => Promise<void>;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  engines,
  isSubmitting,
  onRefreshEngines,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [spaceName, setSpaceName] = useState('');
  const [knownRoomWidth, setKnownRoomWidth] = useState('');
  const [acceptEstimatedWidth, setAcceptEstimatedWidth] = useState(false);
  const [selectedEngine, setSelectedEngine] = useState('photographic');
  const [photographic, setPhotographic] = useState(true);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [captureMode,setCaptureMode] = useState<'photos'|'video'>('photos');
  const [refreshingRuntime,setRefreshingRuntime] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const closeRef = useRef(onClose);
  const submittingRef = useRef(isSubmitting);
  closeRef.current = onClose;
  submittingRef.current = isSubmitting;

  useEffect(() => {
    if (!isOpen) return;
    previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const frame = window.requestAnimationFrame(() => dialogRef.current?.querySelector<HTMLInputElement>('#space-title')?.focus());
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !submittingRef.current) {
        event.preventDefault();
        closeRef.current();
      }
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]):not([type="hidden"]), [role="button"][tabindex="0"], select:not([disabled]), textarea:not([disabled])'))
        .filter(element => element.getClientRects().length > 0);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const outside = !dialogRef.current.contains(document.activeElement);
      if (event.shiftKey && (document.activeElement === first || outside)) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || outside)) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener('keydown', handleKeyDown);
      previousFocusRef.current?.focus();
    };
  }, [isOpen]);

  // A finished or cancelled upload must never remain preloaded when the modal
  // is opened for the next room. This also lets the same filename be chosen
  // again after its contents have changed.
  useEffect(() => {
    if (!isOpen) {
      setSelectedFiles([]);
      setSpaceName('');
      setKnownRoomWidth('');
      setAcceptEstimatedWidth(false);
      setSelectedEngine('photographic');
      setPhotographic(true);
      setError(null);
      setCaptureMode('photos');
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [isOpen]);

  const isVggt = selectedEngine === 'vggt_experimental';
  const isJointExperiment = isVggt || selectedEngine === 'resplat_experimental';
  const unmeasuredCapture = selectedEngine === 'photographic' || isJointExperiment;
  const selectedEngineInfo = engines.find(engine => engine.id === selectedEngine);
  const vggtUnavailable = isJointExperiment && selectedEngineInfo?.can_run !== true;
  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesSelected(Array.from(e.dataTransfer.files));
    }
  };

  const handleFilesSelected = (files: File[]) => {
    setError(null);
    if (files.length === 0) return;
    if(files.reduce((sum,file)=>sum+file.size,0)>=500*1024*1024){
      setError('Keep the complete upload below 500 MB.');return;
    }

    const videos = files.filter(f => /\.(mp4|mov|avi|mkv|webm)$/i.test(f.name));
    if (videos.length) {
      if (videos.length !== 1 || files.length !== 1) {
        setError('Choose either overlapping photos or one walkthrough video.');
        return;
      }
      setSelectedEngine('photographic');
      setCaptureMode('video');
      setSelectedFiles(videos);
      if (!spaceName) setSpaceName(videos[0].name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' '));
      return;
    }

    const photos = files.filter(f => f.type.startsWith('image/') || /\.(jpg|jpeg|png|webp|bmp|tiff)$/i.test(f.name));
    if (photos.length !== files.length) {
      setError('Use only room images: JPG, PNG, WEBP, BMP, or TIFF.');
      return;
    }
    if (photos.length > 60) {
      setError('Choose up to 60 overlapping room photos.');
      return;
    }
    setSelectedFiles(photos);
    setCaptureMode('photos');
    if (!spaceName) {
      setSpaceName(photos[0].name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' '));
    }
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedFiles.length === 0) {
      setError('Choose overlapping room photos or one walkthrough video.');
      return;
    }
    const isVideo = /\.(mp4|mov|avi|mkv|webm)$/i.test(selectedFiles[0].name);
    if (isVideo && selectedEngine !== 'photographic') {
      setError('Choose Photographic room to reconstruct a walkthrough video.');
      return;
    }
    if (selectedEngine === 'photographic' && !isVideo && selectedFiles.length < 3) {
      setError('At least 3 distinct overlapping photos are required. We recommend 5–8 photos from different positions, up to 10 for extra coverage.');
      return;
    }
    const width = Number(knownRoomWidth);
    if (isJointExperiment && (selectedFiles.length < 5 || selectedFiles.length > 10)) {
      setError('Experimental mode requires 5–10 distinct overlapping photos.'); return;
    }
    if (vggtUnavailable) {
      setError(selectedEngineInfo?.availability_message || 'VGGT requires a configured CPU/CUDA runtime with enough memory. No server is started automatically.'); return;
    }
    if (!unmeasuredCapture && !knownRoomWidth && !acceptEstimatedWidth) {
      setError('Enter the measured width, or explicitly choose estimated dimensions below.');
      return;
    }
    if (!unmeasuredCapture && knownRoomWidth && (!Number.isFinite(width) || width < 2 || width > 20)) {
      setError('Known room width must be between 2 and 20 metres.');
      return;
    }
    onSubmit(
      selectedFiles,
      spaceName.trim() || 'Custom 3D Room Space',
      selectedEngine,
      { ...(!unmeasuredCapture && width > 0 ? { knownRoomWidthM: width } : {}), ...(selectedEngine === 'groq_scene' ? { photographic } : {}) },
    );
  };

  return (
    <div data-lenis-prevent className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-xl font-['Inter',sans-serif]">
      <div ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="create-space-title"
        className="relative w-full max-w-2xl bg-[#171412] border border-[#e6d5bd]/15 rounded-3xl shadow-2xl shadow-black/40 overflow-hidden flex flex-col max-h-[90vh]"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#d5360c]/15 flex items-center justify-center text-[#e6d5bd]">
              <Upload className="w-4 h-4" />
            </div>
            <div>
              <h2 id="create-space-title" className="text-base font-bold text-white tracking-wide">Create 3D Space</h2>
              <p className="text-xs text-gray-400">{isVggt ? 'COLMAP + VGGT joint camera/depth preview. Local CPU or CUDA; distortion is not verified fixed. No automatic cloud start.' : selectedEngine === 'photographic' ? 'Overlapping photos → photographic 3D. No Groq or Gemini requests. Scale is unmeasured.' : selectedEngine === 'groq_scene' ? 'Groq analyzes your photos to estimate an editable room.' : selectedEngine === 'gemini_scene' ? 'Gemini analyzes your photos to estimate an editable room.' : 'Local image analysis with estimated dimensions.'}</p>
            </div>
          </div>
          <button
            type="button"
            aria-label="Close create space dialog"
            onClick={onClose}
            disabled={isSubmitting}
            className="w-8 h-8 rounded-full flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto space-y-6">
          {/* Space Name Input */}
          <div>
            <label htmlFor="space-title" className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
              Space Title
            </label>
            <input
              id="space-title"
              type="text"
              value={spaceName}
              onChange={e => setSpaceName(e.target.value)}
              placeholder="e.g. Master Bedroom, Studio Loft, Living Room"
              className="w-full px-4 py-3 rounded-xl bg-white/[0.04] border border-white/10 text-white placeholder-gray-500 text-sm focus:outline-none focus:border-[#d5360c] transition-colors"
            />
          </div>

          {/* Media Dropzone */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300">
                Room Capture ({selectedFiles.length} selected)
              </label>
            </div>

            <div
              role="button"
              tabIndex={0}
              aria-label="Choose room photos or a walkthrough video"
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInputRef.current?.click(); } }}
              className={`relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-[#d5360c] bg-[#d5360c]/10'
                  : 'border-white/15 hover:border-[#d5360c]/40 hover:bg-white/[0.02]'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple={captureMode==='photos'}
                accept={captureMode==='video'?'.mp4,.mov,.avi,.mkv,.webm':'.jpg,.jpeg,.png,.webp,.bmp,.tiff'}
                onChange={e => {
                  if (e.target.files) handleFilesSelected(Array.from(e.target.files));
                  e.target.value = '';
                }}
                className="hidden"
              />

              <div className="flex flex-col items-center justify-center gap-3">
                <div className="w-12 h-12 rounded-2xl bg-[#d5360c]/15 border border-[#e6d5bd]/15 flex items-center justify-center text-[#e6d5bd]">
                  <ImageIcon className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">
                    {isVggt ? 'Add 5–10 distinct overlapping photos' : selectedEngine === 'photographic' ? (captureMode==='video'?'Add one walkthrough video':'Add 5–8 overlapping photos') : 'Add overlapping photos of the same room'}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {isVggt ? 'VGGT evaluates all 5–10 selected photos together. Use translated, overlapping views with unchanged lighting and furniture. This experimental option is photos-only; scale and unseen surfaces remain uncertain.' : 'Use 5–8 sharp photos from different positions; up to 10 is recommended for extra coverage. More are optional (3–60 accepted). Or upload a slow walkthrough, up to 3 minutes. Keep lighting, zoom and furniture unchanged. Unseen surfaces remain uncertain.'}
                  </p>
                </div>
              </div>
            </div>

            {selectedEngine==='photographic'&&<div className="mt-3 flex flex-wrap gap-2" aria-label="Capture type">
              {(['photos','video'] as const).map(mode=><button key={mode} type="button" aria-pressed={captureMode===mode}
                onClick={()=>{setCaptureMode(mode);setSelectedFiles([]);setError(null);if(fileInputRef.current)fileInputRef.current.value='';}}
                className={`rounded-full border px-4 py-2 text-xs ${captureMode===mode?'border-orange-500 bg-orange-500/10 text-white':'border-white/15 text-gray-400'}`}>
                {mode==='video'?'Walkthrough video':'Overlapping photos'}
              </button>)}
              <p className="w-full text-xs text-gray-400">{captureMode==='video'?'MP4, MOV, WEBM, MKV or AVI · 1–180 seconds · below 500 MB. Frames are extracted automatically; audio is not used.':'JPG, PNG, WEBP, BMP or TIFF · 3–60 distinct overlapping views.'}</p>
            </div>}

            {/* Selected File Previews */}
            {selectedFiles.length > 0 && (
              <div className="mt-4">
                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                  <span>Selected files ({selectedFiles.length})</span>
                  <button
                    type="button"
                    onClick={() => setSelectedFiles([])}
                    className="text-red-400 hover:underline"
                  >
                    Clear all
                  </button>
                </div>

                <div className="grid grid-cols-3 sm:grid-cols-4 gap-2 max-h-40 overflow-y-auto p-1">
                  {selectedFiles.map((file, idx) => (
                    <div
                      key={idx}
                      className="relative group rounded-lg overflow-hidden border border-white/10 bg-white/[0.03] p-2 flex flex-col justify-between aspect-video text-left"
                    >
                      <span className="text-[11px] font-medium text-white truncate w-full">
                        {file.name}
                      </span>
                      <span className="text-[10px] text-gray-400 font-mono">
                        {(file.size / (1024 * 1024)).toFixed(1)} MB
                      </span>
                      <button
                        type="button"
                        aria-label={`Remove ${file.name}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveFile(idx);
                        }}
                        className="absolute top-1 right-1 w-5 h-5 rounded-full bg-black/70 text-gray-300 hover:text-white flex items-center justify-center opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus:opacity-100 transition-opacity"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {!unmeasuredCapture && <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
              Room Width — Measured or Estimated
            </label>
            <div className="flex items-center gap-3">
              <input
                type="number"
                min="2"
                max="20"
                step="0.01"
                value={knownRoomWidth}
                onChange={e => setKnownRoomWidth(e.target.value)}
                placeholder="e.g. 4.20"
                className="w-44 px-4 py-3 rounded-xl bg-white/[0.04] border border-white/10 text-white placeholder-gray-500 text-sm focus:outline-none focus:border-[#d5360c] transition-colors"
              />
              <span className="text-xs text-gray-400">metres — calibrates all generated object sizes</span>
            </div>
            {!knownRoomWidth && <label className="flex items-center gap-2 mt-3 text-sm text-amber-200">
              <input type="checkbox" checked={acceptEstimatedWidth}
                onChange={e => setAcceptEstimatedWidth(e.target.checked)} />
              Estimate dimensions from the photo (not an exact measurement).
            </label>}
          </div>}

          {/* Engine Selection */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-[#e6d5bd]" />
              <span>3D Reconstruction Engine (Modular)</span>
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {engines.filter(eng => eng.id !== 'sample_architectural').map(eng => (
                <div
                  key={eng.id}
                  role="button" tabIndex={0} aria-pressed={selectedEngine === eng.id}
                  onClick={() => {setSelectedEngine(eng.id);setError(null);}}
                  onKeyDown={event => {if(event.key==='Enter'||event.key===' '){event.preventDefault();setSelectedEngine(eng.id);setError(null);}}}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    selectedEngine === eng.id
                      ? 'bg-[#d5360c]/10 border-[#d5360c] shadow-md shadow-[#d5360c]/10'
                      : 'bg-white/[0.02] border-white/10 hover:border-white/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold text-white">{eng.name}</span>
                    {eng.is_default && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#d5360c]/20 text-[#e6d5bd] font-mono">
                        IMAGE-DRIVEN
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-gray-400 leading-snug">{eng.description}</p>
                  {eng.experimental && <p className="mt-2 text-[11px] text-amber-200">Experimental · {eng.can_run ? 'Device checked when started' : eng.availability_code === 'vggt_local_memory_low' ? 'Installed locally · free RAM needed' : 'Runtime needs attention'}</p>}
                </div>
              ))}
            </div>
          </div>

          {isJointExperiment && <div role="status" className="rounded-xl border border-amber-400/30 p-3 text-xs text-amber-100">
            <p>{selectedEngineInfo?.availability_message || 'Requires a configured CPU/CUDA runtime and sufficient memory.'}</p>
            {onRefreshEngines && <button type="button" disabled={refreshingRuntime} className="mt-2 underline disabled:opacity-50" onClick={async()=>{
              setRefreshingRuntime(true);
              try { await onRefreshEngines(); setError(null); }
              catch { setError('Could not recheck the local runtime. No reconstruction was started.'); }
              finally { setRefreshingRuntime(false); }
            }}>{refreshingRuntime?'Checking local runtime…':'Recheck local memory'}</button>}
            <p className="mt-2">Use 5–10 distinct overlapping photos. Video stays on Photographic room. A successful experimental run opens an unverified preview for review—not a finished room in Recents. Scale and unseen walls remain estimates.</p>
          </div>}

          {selectedEngine === 'groq_scene' && <label className="block rounded-xl border border-white/10 p-3 text-sm text-gray-200">
            <input type="checkbox" checked={photographic} onChange={event => setPhotographic(event.target.checked)} className="mr-2" />
            Add photographic reconstruction (COLMAP + Brush)
            <span className="mt-2 block text-xs text-gray-400">Requires 3+ distinct overlapping photos and a compatible backend GPU. Adds several minutes. The editable room remains available if training fails. Unseen surfaces and accurate scale are not guaranteed.</span>
          </label>}
          {error && (
            <div role="alert" className="flex items-center gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-xs">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit Action */}
          <div className="pt-2 flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-5 py-2.5 rounded-xl text-xs font-medium text-gray-300 hover:text-white hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || selectedFiles.length === 0 || vggtUnavailable || refreshingRuntime}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-[#d5360c] hover:bg-[#b82e0a] disabled:opacity-50 text-white font-semibold text-xs shadow-lg shadow-[#d5360c]/20 transition-all cursor-pointer"
            >
              {isSubmitting ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Uploading...</span>
                </>
              ) : (
                <>
                  <span>Start 3D Reconstruction</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
