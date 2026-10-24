import React, { useEffect, useRef } from 'react';
import { Sparkles, Terminal, CheckCircle2, AlertCircle, ArrowRight, Box, Cpu, Layers } from 'lucide-react';
import { SpaceJob } from '../types';

interface ProcessingViewProps {
  job: SpaceJob;
  onEnterViewer: () => void;
  onCancel: () => void;
  onRetry: () => void;
  isRetrying: boolean;
}

const STAGES = [
  { id: 1, label: 'Load Room Photo', minPct: 5 },
  { id: 2, label: 'Analyze Room Structure', minPct: 20 },
  { id: 3, label: 'Identify Visible Objects', minPct: 30 },
  { id: 4, label: 'SAM 2 + MoGe Placement', minPct: 54 },
  { id: 5, label: 'Constrained GLB Compilation', minPct: 80 },
];
const PHOTOGRAPHIC_STAGES = [
  {id:1,label:'Load Photos / Video',minPct:5},
  {id:2,label:'COLMAP + Brush Training',minPct:35},
  {id:3,label:'Photographic Refinement',minPct:70},
  {id:4,label:'Depth, Walls + Collisions',minPct:90},
  {id:5,label:'Publish Walkthrough',minPct:100},
];
const VGGT_STAGES = [
  {id:1,label:'Check Local Runtime',minPct:5},
  {id:2,label:'COLMAP / Joint Inputs',minPct:15},
  {id:3,label:'VGGT Camera + Depth',minPct:30},
  {id:4,label:'Validate Static Preview',minPct:75},
  {id:5,label:'Review — Not Published',minPct:100},
];
const RESPLAT_STAGES = [
  {id:1,label:'Check CUDA Runtime',minPct:5},
  {id:2,label:'COLMAP Cameras',minPct:15},
  {id:3,label:'ReSplat Joint Refinement',minPct:35},
  {id:4,label:'Validate Gaussian Output',minPct:90},
  {id:5,label:'Moving-View Review',minPct:100},
];

export const ProcessingView: React.FC<ProcessingViewProps> = ({
  job,
  onEnterViewer,
  onCancel,
  onRetry,
  isRetrying,
}) => {
  const terminalBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    terminalBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [job.logs]);

  const isCompleted = job.status === 'completed';
  const isReview = job.status === 'review_required';
  const isFailed = job.status === 'failed';
  const usedLocalFallback = job.metrics?.extra_details?.lighting?.analysis_quality === 'local_fallback';
  const analysisWarning = job.metrics?.extra_details?.lighting?.analysis_warning;
  const fineShapesOnly = typeof analysisWarning === 'string' && /^Fine shapes use reusable approximate geometry: [\w, ]+\.$/.test(analysisWarning.trim());
  const alignment = job.metrics?.extra_details?.self_correction;
  const reviewObjects = alignment?.objects?.filter((item: { needs_review: boolean }) => item.needs_review) ?? [];

  return (
    <div className="processing-brand-view relative min-h-screen pt-24 pb-16 px-6 flex flex-col justify-center items-center overflow-hidden bg-[#101010] font-['Inter',sans-serif]">
      {/* Background Spatial Glows */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[550px] bg-[#d5360c]/10 rounded-full blur-[140px] pointer-events-none" />

      <div className="relative z-10 w-full max-w-3xl mx-auto flex flex-col items-center text-center">
        {/* Spatial Radar / Scanner Animation */}
        <div className="relative w-36 h-36 mb-8 flex items-center justify-center">
          <div className="absolute inset-0 rounded-full border border-[#d5360c]/20 animate-ping opacity-25" />
          <div className="absolute inset-2 rounded-full border border-[#d5360c]/40 animate-spin-slow" />
          <div className="absolute inset-6 rounded-full border-t-2 border-[#e6d5bd] animate-spin" />
          <div className="w-20 h-20 rounded-full bg-[#d5360c]/10 backdrop-blur-md border border-[#d5360c]/40 flex items-center justify-center shadow-lg shadow-[#d5360c]/30">
            {isCompleted ? (
              <CheckCircle2 className="w-10 h-10 text-[#e6d5bd]" />
            ) : isFailed ? (
              <AlertCircle className="w-10 h-10 text-red-400" />
            ) : (
              <Box className="w-8 h-8 text-[#e6d5bd] animate-pulse" />
            )}
          </div>
        </div>

        {/* Status Heading */}
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white mb-2">
          {isReview ? `${job.engine_id==='resplat_experimental'?'ReSplat':'VGGT'} Experimental Preview — Review Required` : isCompleted
            ? (analysisWarning ? (fineShapesOnly ? '3D Room Ready — Approximate Details' : 'Partial 3D Room — Review Needed') : '3D Spatial Room Ready')
            : isFailed
            ? (job.error_code === 'analysis_quota_exhausted' ? 'Photo Analysis Daily Quota Reached' : job.error_code === 'analysis_service_unavailable' ? 'Photo Analysis Service Is Busy' : job.error_code === 'analysis_incomplete' ? 'Full Reconstruction Not Verified' : 'Reconstruction Encountered An Error')
            : 'Reconstructing 3D Spatial Space'}
        </h2>
        <p className="text-sm text-gray-400 max-w-md mb-8">
          {isReview ? 'This preview is unverified and is not in Recents. Inspect translated views and Geometry check; distortion, hidden surfaces and scale are not confirmed accurate.' : isCompleted
            ? (job.gaussian?.status === 'ready' ? (job.glb_url?'Photographic and editable room views are ready. Hidden surfaces and accurate scale remain unverified.':'Your photographic walkthrough is ready. Unseen wall completion and collision geometry are estimates; scale is unmeasured.') : 'Your estimated room layout is ready as separate 3D objects. Sizes and depths are inferred, not measured.')
            : isFailed
            ? (job.error || 'Failed to process spatial media')
            : job.stage || 'Analyzing the photo and estimating the visible room layout...'}
        </p>

        {isCompleted && job.gaussian?.reason && <div role="status" className="mb-6 max-w-xl rounded-xl border border-white/15 p-4 text-left text-sm text-gray-300">{job.gaussian.reason}</div>}
        {isCompleted && analysisWarning && (
          <div role="alert" className="w-full max-w-xl rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 mb-6 text-left text-sm text-amber-100">
            <strong>{fineShapesOnly ? 'Some fine shapes are approximate.' : 'Object detection was not fully verified.'}</strong> {analysisWarning}
            <p className="mt-2">{fineShapesOnly ? 'Reconstruction completed, but these objects use reusable geometry rather than verified photo-specific shapes. This does not establish exact photo accuracy; review the alignment check below.' : 'This is a partial result, not a complete reconstruction of the photo. Retry reconstruction when the analysis service is available.'}</p>
          </div>
        )}
        {isCompleted && alignment && (
          <div className="w-full max-w-xl rounded-xl border border-[#d5360c]/25 bg-[#d5360c]/5 p-4 mb-6 text-left text-sm text-gray-200">
            <h3 className="font-semibold text-white mb-2">Editable room photo alignment check</h3>
            {typeof alignment.before_iou === 'number' && typeof alignment.after_iou === 'number' ? (
              <>
                <p>Detected-mask overlap: {alignment.before_iou.toFixed(2)} → {alignment.after_iou.toFixed(2)} (0–1).</p>
                <p>{alignment.status === 'corrected' ? 'Safe corrections were kept and the exported model was rechecked.' : 'The original layout was kept because no verified improvement was available.'}</p>
                {reviewObjects.length > 0 && <p className="text-amber-200 mt-2">Check these objects: {reviewObjects.map((item: { label: string }) => item.label.replace(/_/g, ' ')).join(', ')}.</p>}
              </>
            ) : <p>The alignment check was unavailable. This room has not been verified against the photo masks.</p>}
            <p className="text-gray-400 mt-2">This measures visible silhouette agreement—not exact 3D accuracy. Missed detections and hidden surfaces are not covered.</p>
          </div>
        )}

        {/* Progress Bar */}
        <div className="w-full max-w-xl mb-10">
          <div className="flex items-center justify-between text-xs font-mono mb-2">
            <span className="text-[#e6d5bd] font-semibold tracking-wider flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" />
              <span>PIPELINE TELEMETRY</span>
            </span>
            <span className="text-white font-bold">{Math.round(job.progress || 0)}%</span>
          </div>
          <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden p-[1px]">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#d5360c] to-[#e6d5bd] shadow-sm shadow-[#d5360c] transition-all duration-300 ease-out"
              style={{ width: `${Math.max(5, job.progress || 5)}%` }}
            />
          </div>
        </div>

        {/* Pipeline Stepper */}
        <div className="w-full max-w-xl grid grid-cols-1 sm:grid-cols-5 gap-2 mb-8">
          {(job.engine_id==='resplat_experimental'?RESPLAT_STAGES:job.engine_id==='vggt_experimental'?VGGT_STAGES:job.engine_id==='photographic'?PHOTOGRAPHIC_STAGES:STAGES).map((s) => {
            const isDone = (job.progress || 0) >= s.minPct;
            const isCurrent = !isDone && (job.progress || 0) >= (s.minPct - 20);
            return (
              <div
                key={s.id}
                className={`p-2 rounded-xl text-left border transition-all ${
                  isDone
                    ? 'bg-[#d5360c]/10 border-[#d5360c]/30 text-[#e6d5bd]'
                    : isCurrent
                    ? 'bg-white/[0.04] border-white/20 text-white'
                    : 'bg-white/[0.01] border-white/5 text-gray-500'
                }`}
              >
                <div className="text-[10px] font-mono mb-0.5">STAGE 0{s.id}</div>
                <div className="text-[11px] font-medium truncate">{s.label}</div>
              </div>
            );
          })}
        </div>

        {/* Live Terminal Telemetry Logs */}
        <div className="w-full max-w-xl text-left mb-8">
          <div className="flex items-center justify-between px-4 py-2 bg-[#1b1815] rounded-t-xl border-t border-x border-white/10">
            <div className="flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-[#e6d5bd]" />
              <span className="text-[11px] font-mono text-gray-300">Live Execution Log</span>
            </div>
            <span className="text-[10px] font-mono text-gray-500">
              {job.input_type.toUpperCase()} INGESTION
            </span>
          </div>
          <div className="h-36 overflow-y-auto bg-[#0b0b0b] p-3 rounded-b-xl border border-white/10 font-mono text-xs text-gray-400 space-y-1">
            {job.logs.map((log, i) => (
              <div key={i} className="flex gap-2">
                <span className="text-[#d5360c] select-none">&gt;</span>
                <span className="text-gray-300">{log}</span>
              </div>
            ))}
            <div ref={terminalBottomRef} />
          </div>
        </div>

        {/* Metrics Card upon completion */}
        {isCompleted && usedLocalFallback && (
          <div className="w-full max-w-xl mb-4 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-left text-xs text-amber-100">
            Gemini was unavailable for this run, so the result used lower-accuracy local image analysis. Re-run the photo for precise object placement.
          </div>
        )}

        {isCompleted && job.metrics && (
          <div className="w-full max-w-xl glass-panel p-4 rounded-2xl mb-8 text-left grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <div className="text-[10px] text-gray-400 font-mono uppercase">Room Dimensions</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {job.engine_id === 'photographic' ? 'Unmeasured capture' : `${job.metrics.estimated_room_width}m × ${job.metrics.estimated_room_length}m`}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-gray-400 font-mono uppercase">Room Height</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {job.engine_id === 'photographic' ? 'Unmeasured' : `${job.metrics.estimated_room_height}m`}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-gray-400 font-mono uppercase">3D Point Cloud</div>
              <div className="text-sm font-bold text-[#e6d5bd] mt-0.5">
                {job.metrics.points_3d_triangulated.toLocaleString()} pts
              </div>
            </div>
            <div>
              <div className="text-[10px] text-gray-400 font-mono uppercase">Processing Time</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {job.metrics.processing_duration_sec}s
              </div>
            </div>
          </div>
        )}

        {/* Primary Action Button */}
        {isCompleted || isReview ? (
          <button
            onClick={onEnterViewer}
            className="flex items-center gap-3 px-8 py-4 rounded-2xl bg-[#d5360c] hover:bg-[#b82e0a] text-white font-bold text-sm shadow-xl shadow-[#d5360c]/30 hover:scale-[1.03] active:scale-[0.98] transition-all cursor-pointer"
          >
            <Sparkles className="w-4 h-4" />
            <span>{isReview ? 'Inspect Experimental Preview' : 'Enter Immersive 3D Space'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        ) : isFailed ? (
          <button
            onClick={onRetry}
            disabled={isRetrying}
            className="px-6 py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium text-sm transition-colors"
          >
            {isRetrying ? 'Starting retry...' : 'Retry Same Upload'}
          </button>
        ) : (
          <button
            onClick={onCancel}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            Cancel reconstruction and return home
          </button>
        )}
      </div>
    </div>
  );
};
