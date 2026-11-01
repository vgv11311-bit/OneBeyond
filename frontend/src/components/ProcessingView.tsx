import React, { useEffect, useRef } from 'react';
import { Sparkles, Terminal, CheckCircle2, AlertCircle, ArrowRight, Box, Cpu, Layers } from 'lucide-react';
import { SpaceJob } from '../types';

interface ProcessingViewProps {
  job: SpaceJob;
  onEnterViewer: () => void;
  onCancel: () => void;
  onRetry: () => void;
  isRetrying: boolean;
  connectionWarning?: string;
}

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
  connectionWarning,
}) => {
  const terminalBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    terminalBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [job.logs]);

  const isCompleted = job.status === 'completed';
  const isReview = job.status === 'review_required';
  const isCancelled=job.status==='cancelled';
  const isFailed = job.status === 'failed' || isCancelled;
  const isTerminal=isCompleted||isReview||isFailed;
  const activity=job.status==='processing'?job.runtime_activity:undefined;
  const duration=(seconds:number)=>`${Math.floor(seconds/60)}m ${Math.floor(seconds%60)}s`;
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
          <div className={`absolute inset-0 rounded-full border border-[#d5360c]/20 opacity-25 ${isTerminal?'':'animate-ping'}`} />
          <div className={`absolute inset-2 rounded-full border border-[#d5360c]/40 ${isTerminal?'':'animate-spin-slow'}`} />
          <div className={`absolute inset-6 rounded-full border-t-2 border-[#e6d5bd] ${isTerminal?'':'animate-spin'}`} />
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
          {isCancelled ? 'Reconstruction Cancelled' : isReview ? `$ReSplat Experimental Preview — Review Required` : isCompleted
            ? (analysisWarning ? (fineShapesOnly ? '3D room ready — approximate details' : 'Partial 3D room — review needed') : '3D room ready to inspect')
            : isFailed
            ? (job.error_code === 'analysis_quota_exhausted' ? 'Photo Analysis Daily Quota Reached' : job.error_code === 'analysis_service_unavailable' ? 'Photo Analysis Service Is Busy' : job.error_code === 'analysis_incomplete' ? 'Full Reconstruction Not Verified' : 'Reconstruction could not finish')
            : 'Reconstructing your room'}
        </h2>
        <p className="text-sm text-gray-400 max-w-md mb-8">
          {isCancelled ? 'Processing has stopped. Your upload is saved; use Retry reconstruction to start another attempt.' : isReview ? job.gaussian?.saved_to_recents ? 'This preview is saved in Your Spaces for review, but quality has not been verified. Move to different viewpoints to inspect the geometry; distortion, hidden surfaces and scale may be inaccurate.' : 'This preview is unverified and is not in Your Spaces. Move to different viewpoints and inspect the geometry. Distortion, hidden surfaces and scale may be inaccurate.' : isCompleted
            ? (job.gaussian?.status === 'ready' ? (job.glb_url?'Photographic and editable room views are ready. Hidden surfaces and accurate scale remain unverified.':'Your photographic walkthrough is ready. Unseen wall completion and collision geometry are estimates; scale is unmeasured.') : 'Your estimated room layout is ready as separate 3D objects. Sizes and depths are inferred, not measured.')
            : isFailed
            ? (job.error || 'We could not process this capture. Please try again.')
            : job.stage || 'Analyzing your capture and estimating the visible room layout…'}
        </p>

        {connectionWarning && <div role="alert" className="mb-6 w-full max-w-xl rounded-xl border border-amber-500/40 p-4 text-left text-sm text-amber-100">{connectionWarning} No new progress is confirmed until the connection returns.</div>}
        {!isTerminal && job.processing_budget && <div role="status" className="mb-6 w-full max-w-xl rounded-xl border border-white/15 p-4 text-left text-sm text-gray-300">
          <p>Processing: {duration(job.processing_budget.elapsed_seconds)} · five-minute benchmark: {duration(job.processing_budget.remaining_seconds)} remaining</p>
          <p>Preview target: 3 minutes · completed-room benchmark: 5 minutes.</p>
          {job.processing_budget.deadline_exceeded ? <p>The five-minute benchmark was missed. Processing continues; a completed room is not yet confirmed.</p> : job.processing_budget.soft_target_exceeded && <p>The preview target was missed. Validation is still required; a completed room is not yet confirmed.</p>}
          <p className="mt-1 text-xs text-gray-400">Upload, queue and GPU startup are separate. The benchmark does not stop processing or publish an unfinished room. Your GPU session and safety limits still apply.</p>
        </div>}
        {activity && <div role="status" className="mb-6 w-full max-w-xl rounded-xl border border-white/15 p-4 text-left text-sm text-gray-300">
          <strong>{activity.label}</strong>
          {activity.total && <p>{activity.current} / {activity.total} {activity.unit}</p>}
          <p>Current step: {duration(activity.elapsed_seconds)} · step time limit remaining: {duration(activity.remaining_seconds)}</p>
          <p>{activity.heartbeat_stale ? 'Worker heartbeat is delayed; completion is not confirmed.' : `Worker heartbeat received ${activity.heartbeat_age_seconds}s ago.`}</p>
          {activity.output_age_seconds>90 && <p>The solver has not written new output for {duration(activity.output_age_seconds)}. It remains bounded by the step time limit.</p>}
          <p className="mt-1 text-xs text-gray-400">The percentage estimates pipeline stages. Frame and matching counters report actual processing work.</p>
        </div>}

        {isCompleted && job.gaussian?.reason && <div role="status" className="mb-6 max-w-xl rounded-xl border border-white/15 p-4 text-left text-sm text-gray-300">{job.gaussian.reason}</div>}
        {isCompleted && analysisWarning && (
          <div role="alert" className="w-full max-w-xl rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 mb-6 text-left text-sm text-amber-100">
            <strong>{fineShapesOnly ? 'Some fine shapes are approximate.' : 'Object detection was not fully verified.'}</strong> {analysisWarning}
            <p className="mt-2">{fineShapesOnly ? 'Reconstruction completed, but these objects use reusable geometry rather than verified photo-specific shapes. This does not establish exact photo accuracy; review the alignment check below.' : 'This is a partial result, not a complete reconstruction of the photo. Retry reconstruction when the analysis service is available.'}</p>
          </div>
        )}
        {isCompleted && alignment && (
          <div className="w-full max-w-xl rounded-xl border border-[#d5360c]/25 bg-[#d5360c]/5 p-4 mb-6 text-left text-sm text-gray-200">
            <h3 className="font-semibold text-white mb-2">Photo alignment check</h3>
            {typeof alignment.before_iou === 'number' && typeof alignment.after_iou === 'number' ? (
              <>
                <p>Detected-shape overlap: {alignment.before_iou.toFixed(2)} → {alignment.after_iou.toFixed(2)} (0–1).</p>
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
          {RESPLAT_STAGES.map((s) => {
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

        {isCompleted && job.metrics && (
          <div className="w-full max-w-xl glass-panel p-4 rounded-2xl mb-8 text-left grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <div className="text-[10px] text-gray-400 font-mono uppercase">Estimated room dimensions</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {job.engine_id === 'resplat_experimental' ? 'Unmeasured capture' : `${job.metrics.estimated_room_width}m × ${job.metrics.estimated_room_length}m`}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-gray-400 font-mono uppercase">Estimated room height</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {job.engine_id === 'resplat_experimental' ? 'Unmeasured' : `${job.metrics.estimated_room_height}m`}
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
            <span>{isReview ? 'Inspect Experimental Preview' : 'Explore 3D room'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        ) : isFailed ? (
          <button
            onClick={onRetry}
            disabled={isRetrying}
            className="px-6 py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white font-medium text-sm transition-colors"
          >
            {isRetrying ? 'Starting retry...' : 'Retry reconstruction'}
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
