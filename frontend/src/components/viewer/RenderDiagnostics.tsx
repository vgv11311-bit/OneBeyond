import React, { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { RenderMetrics } from '../../types';

interface RenderDiagnosticsProps {
  modelLoadMs?: number;
  onMetrics: (metrics: RenderMetrics) => void;
}

type PerformanceWithMemory = Performance & {
  memory?: { usedJSHeapSize: number };
};

export const RenderDiagnostics: React.FC<RenderDiagnosticsProps> = ({ modelLoadMs, onMetrics }) => {
  const { gl } = useThree();
  const windowStart = useRef(performance.now());
  const frameDurations = useRef<number[]>([]);
  const renderSize = useRef(new THREE.Vector2());

  useFrame((_, delta) => {
    const now = performance.now();
    // Background-tab throttling is not a renderer performance sample.
    if (document.hidden || delta > 1) {
      frameDurations.current = [];
      windowStart.current = now;
      return;
    }
    frameDurations.current.push(delta * 1000);
    const elapsed = now - windowStart.current;
    if (elapsed < 1000) return;

    const samples = frameDurations.current;
    const sorted = [...samples].sort((a, b) => a - b);
    const total = samples.reduce((sum, value) => sum + value, 0);
    const p95Index = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
    const renderer = gl as any;
    renderer.getDrawingBufferSize(renderSize.current);
    let backend: RenderMetrics['backend'];
    if (renderer.isWebGPURenderer) {
      backend = renderer.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGPU → WebGL2';
    } else {
      const context = renderer.getContext();
      const isWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && context instanceof WebGL2RenderingContext;
      backend = isWebGL2 ? 'WebGL2' : 'WebGL1';
    }
    const rendererInfo = renderer.info;
    const frameWork = renderer.isWebGPURenderer && renderer.domElement.dataset.frameWork
      ? JSON.parse(renderer.domElement.dataset.frameWork) : rendererInfo.render;
    const browserPerformance = performance as PerformanceWithMemory;

    const metrics: RenderMetrics = {
      backend,
      fps: Math.round((samples.length * 1000) / elapsed),
      averageFrameMs: Number((total / Math.max(samples.length, 1)).toFixed(2)),
      p95FrameMs: Number((sorted[p95Index] || 0).toFixed(2)),
      drawCalls: frameWork.drawCalls ?? frameWork.calls ?? 0,
      triangles: frameWork.triangles ?? 0,
      geometries: rendererInfo.memory.geometries ?? 0,
      textures: rendererInfo.memory.textures ?? 0,
      renderWidth: Math.round(renderSize.current.x),
      renderHeight: Math.round(renderSize.current.y),
      modelLoadMs,
      heapMemoryMb: browserPerformance.memory
        ? Number((browserPerformance.memory.usedJSHeapSize / 1024 / 1024).toFixed(1))
        : undefined,
    };
    gl.domElement.dataset.renderMetrics = JSON.stringify(metrics);
    onMetrics(metrics);

    frameDurations.current = [];
    windowStart.current = now;
  });

  return null;
};
