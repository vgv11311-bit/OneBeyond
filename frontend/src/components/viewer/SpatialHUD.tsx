import React, { useState, useRef } from 'react';
import { Compass, Eye, Info, Activity, Save, Move } from 'lucide-react';
import {
  GraphicsCapabilities,
  GraphicsQuality,
  RenderMetrics,
  ResolvedGraphicsQuality,
  SpaceJob,
} from '../../types';

interface SpatialHUDProps {
  job: SpaceJob;
  isPointerLocked: boolean;
  onLockPointer: () => void;
  onVirtualMove?: (dx: number, dy: number) => void;
  onVirtualHeight?: (delta: number) => void;
  graphicsQuality: GraphicsQuality;
  resolvedGraphicsQuality: ResolvedGraphicsQuality;
  capabilities?: GraphicsCapabilities;
  renderMetrics?: RenderMetrics;
  baselineCaptured: boolean;
  onCaptureBaseline: () => void;
}

export const SpatialHUD: React.FC<SpatialHUDProps> = ({
  job,
  isPointerLocked,
  onLockPointer,
  onVirtualMove,
  onVirtualHeight,
  graphicsQuality,
  resolvedGraphicsQuality,
  capabilities,
  renderMetrics,
  baselineCaptured,
  onCaptureBaseline,
}) => {
  const [showHelp, setShowHelp] = useState(true);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  // Joystick touch state
  const joystickRef = useRef<HTMLDivElement>(null);
  const [joystickActive, setJoystickActive] = useState(false);
  const [knobPos, setKnobPos] = useState({ x: 0, y: 0 });
  const touchIdRef = useRef<number | null>(null);
  const centerRef = useRef({ x: 0, y: 0 });

  // Virtual Joystick touch handlers
  const handleTouchStart = (e: React.TouchEvent) => {
    if (joystickRef.current && touchIdRef.current === null) {
      const touch = e.changedTouches[0];
      touchIdRef.current = touch.identifier;
      const rect = joystickRef.current.getBoundingClientRect();
      centerRef.current = {
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
      };
      setJoystickActive(true);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (touchIdRef.current !== null) {
      for (let i = 0; i < e.changedTouches.length; i++) {
        const touch = e.changedTouches[i];
        if (touch.identifier === touchIdRef.current) {
          const dx = touch.clientX - centerRef.current.x;
          const dy = touch.clientY - centerRef.current.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const maxRadius = 40;
          const clampedDist = Math.min(dist, maxRadius);
          const angle = Math.atan2(dy, dx);
          
          const knobX = Math.cos(angle) * clampedDist;
          const knobY = Math.sin(angle) * clampedDist;
          
          setKnobPos({ x: knobX, y: knobY });

          if (onVirtualMove) {
            onVirtualMove(knobX / maxRadius, -knobY / maxRadius);
          }
          break;
        }
      }
    }
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    for (let i = 0; i < e.changedTouches.length; i++) {
      if (e.changedTouches[i].identifier === touchIdRef.current) {
        touchIdRef.current = null;
        setJoystickActive(false);
        setKnobPos({ x: 0, y: 0 });
        if (onVirtualMove) onVirtualMove(0, 0);
        break;
      }
    }
  };

  const metrics = job.metrics;

  return (
    <div className="absolute inset-0 pointer-events-none z-30 flex flex-col justify-between p-4 sm:p-6 select-none">
      {/* Top HUD Info */}
      <div className="flex items-start justify-between">
        {/* Room Title & Spatial Specs */}
        <div className="glass-panel p-3 sm:p-4 rounded-2xl pointer-events-auto border border-white/10 max-w-sm">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-[10px] font-mono uppercase tracking-widest text-cyan-300 font-bold">
              ONE BEYOND &bull; SPATIAL ROOM
            </span>
          </div>
          <h3 className="text-sm font-bold text-white truncate">
            {job.name || 'Reconstructed Space'}
          </h3>
          {metrics && (
            <div className="flex items-center gap-3 mt-2 pt-2 border-t border-white/10 text-[11px] font-mono text-gray-300">
              <span>{metrics.estimated_room_width}m × {metrics.estimated_room_length}m</span>
              <span className="text-gray-600">&bull;</span>
              <span>{metrics.estimated_room_height}m H</span>
              <span className="text-gray-600">&bull;</span>
              <span className="text-cyan-400">{metrics.points_3d_triangulated} pts</span>
            </div>
          )}
        </div>

        {/* Spatial renderer stats and active backend diagnostics. */}
        <div className="relative pointer-events-auto">
          <button
            type="button"
            onClick={() => setShowDiagnostics((shown) => !shown)}
            className="glass-panel px-3 py-2 rounded-xl border border-white/10 flex items-center gap-3 text-xs font-mono hover:bg-white/10 transition-colors"
          >
            <div className="flex items-center gap-1.5 text-gray-400">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              <span>{renderMetrics?.backend || (capabilities?.webgpuAvailable ? 'WebGPU' : capabilities?.webgl2Available ? 'WebGL2' : 'WebGL')}</span>
            </div>
            <span className="text-gray-600">|</span>
            <span className={`font-bold ${(renderMetrics?.fps || 0) >= 50 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {renderMetrics?.fps ?? '--'} FPS
            </span>
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
          </button>

          {showDiagnostics && (
            <div className="absolute right-0 top-12 w-72 glass-panel rounded-2xl border border-white/10 p-3.5 text-[11px] font-mono text-gray-300 shadow-2xl">
              <div className="flex items-center justify-between mb-3">
                <span className="text-cyan-300 uppercase tracking-widest font-bold">Render Diagnostics</span>
                <span className="text-gray-500 uppercase">{graphicsQuality} → {resolvedGraphicsQuality}</span>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                <span className="text-gray-500">Active backend</span><span className="text-right">{renderMetrics?.backend || 'Detecting'}</span>
                <span className="text-gray-500">WebGPU</span><span className={`text-right ${capabilities?.webgpuAvailable ? 'text-emerald-400' : 'text-gray-500'}`}>{capabilities?.webgpuAvailable ? 'Available' : 'Unavailable'}</span>
                <span className="text-gray-500">Frame avg / p95</span><span className="text-right">{renderMetrics ? `${renderMetrics.averageFrameMs} / ${renderMetrics.p95FrameMs} ms` : '--'}</span>
                <span className="text-gray-500">Draw calls</span><span className="text-right">{renderMetrics?.drawCalls ?? '--'}</span>
                <span className="text-gray-500">Triangles</span><span className="text-right">{renderMetrics?.triangles.toLocaleString() ?? '--'}</span>
                <span className="text-gray-500">GPU resources</span><span className="text-right">{renderMetrics ? `${renderMetrics.geometries} geo / ${renderMetrics.textures} tex` : '--'}</span>
                <span className="text-gray-500">Render target</span><span className="text-right">{renderMetrics ? `${renderMetrics.renderWidth}×${renderMetrics.renderHeight}` : '--'}</span>
                <span className="text-gray-500">Model load</span><span className="text-right">{renderMetrics?.modelLoadMs ? `${Math.round(renderMetrics.modelLoadMs)} ms` : '--'}</span>
                <span className="text-gray-500">JS heap</span><span className="text-right">{renderMetrics?.heapMemoryMb ? `${renderMetrics.heapMemoryMb} MB` : 'Browser hidden'}</span>
              </div>
              <div className="mt-3 pt-3 border-t border-white/10 text-[10px] text-gray-500">
                {renderMetrics?.backend === 'WebGPU'
                  ? 'Native WebGPU HDR/PBR pipeline active with real shadows and adaptive resolution. High mode adds bloom.'
                  : renderMetrics?.backend === 'WebGPU → WebGL2'
                    ? 'The same HDR/PBR pipeline is active through its WebGL2 fallback.'
                    : 'Legacy WebGL compatibility mode is active for this device.'}
              </div>
              <button
                type="button"
                disabled={!renderMetrics}
                onClick={onCaptureBaseline}
                className="mt-3 w-full flex items-center justify-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-cyan-300 disabled:opacity-40"
              >
                <Save className="w-3.5 h-3.5" />
                {baselineCaptured ? 'Baseline Saved' : 'Save Performance Baseline'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Center Aim Reticle / Pointer Lock Hint */}
      {!isPointerLocked && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-auto text-center">
          <button
            onClick={onLockPointer}
            className="hidden sm:flex items-center gap-2 px-5 py-2.5 rounded-full glass-panel hover:bg-cyan-500/20 hover:border-cyan-400 text-white text-xs font-medium border border-white/20 shadow-xl backdrop-blur-md transition-all cursor-pointer group"
          >
            <Eye className="w-3.5 h-3.5 text-cyan-400 group-hover:scale-110 transition-transform" />
            <span>Click Canvas for First-Person Mouse Look</span>
          </button>
        </div>
      )}

      {/* Center Subtle Crosshair when pointer locked */}
      {isPointerLocked && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/80 shadow-[0_0_8px_#00f0ff]" />
        </div>
      )}

      {/* Bottom Controls Area */}
      <div className="flex items-end justify-between w-full">
        {/* Desktop Controls Card (collapsible) */}
        <div className="hidden sm:block pointer-events-auto">
          {showHelp ? (
            <div className="glass-panel p-3.5 rounded-2xl border border-white/10 text-xs max-w-xs transition-all">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider font-semibold">
                  Controls Guide
                </span>
                <button
                  onClick={() => setShowHelp(false)}
                  className="text-gray-500 hover:text-gray-300 text-[10px]"
                >
                  Hide
                </button>
              </div>
              <div className="grid grid-cols-2 gap-y-1.5 gap-x-3 text-[11px] text-gray-300 font-mono">
                <div className="flex items-center gap-1.5">
                  <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white font-bold text-[10px]">W A S D</kbd>
                  <span className="text-gray-400">Walk</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white font-bold text-[10px]">Mouse</kbd>
                  <span className="text-gray-400">Look</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white font-bold text-[10px]">Shift</kbd>
                  <span className="text-gray-400">Sprint</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-white font-bold text-[10px]">Space / C</kbd>
                  <span className="text-gray-400">Elevation</span>
                </div>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setShowHelp(true)}
              className="glass-panel p-2 rounded-xl text-gray-400 hover:text-white border border-white/10 text-xs flex items-center gap-1.5"
            >
              <Info className="w-3.5 h-3.5 text-cyan-400" />
              <span>Controls</span>
            </button>
          )}
        </div>

        {/* Mobile Virtual Joystick & Touch Controls */}
        <div className="sm:hidden flex items-end justify-between w-full pointer-events-auto">
          {/* Virtual Joystick */}
          <div
            ref={joystickRef}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            className="relative w-28 h-28 rounded-full glass-panel border border-white/20 flex items-center justify-center touch-none"
          >
            <div className="absolute inset-2 rounded-full border border-dashed border-white/20 pointer-events-none" />
            <div
              className={`w-12 h-12 rounded-full bg-cyan-500/40 border border-cyan-400 shadow-lg shadow-cyan-500/30 transition-transform duration-75 flex items-center justify-center ${
                joystickActive ? 'scale-110' : ''
              }`}
              style={{
                transform: `translate(${knobPos.x}px, ${knobPos.y}px)`,
              }}
            >
              <Move className="w-4 h-4 text-white" />
            </div>
          </div>

          {/* Mobile Height Buttons */}
          <div className="flex flex-col gap-2">
            <button
              onTouchStart={() => onVirtualHeight && onVirtualHeight(0.5)}
              className="w-12 h-12 rounded-xl glass-panel border border-white/20 flex items-center justify-center text-white text-xs font-bold active:bg-cyan-500/30"
            >
              ▲ Up
            </button>
            <button
              onTouchStart={() => onVirtualHeight && onVirtualHeight(-0.5)}
              className="w-12 h-12 rounded-xl glass-panel border border-white/20 flex items-center justify-center text-white text-xs font-bold active:bg-cyan-500/30"
            >
              ▼ Down
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
