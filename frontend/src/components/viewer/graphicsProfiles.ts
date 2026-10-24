import {
  GraphicsCapabilities,
  GraphicsQuality,
  RenderMetrics,
  ResolvedGraphicsQuality,
} from '../../types';

export type GraphicsProfile = {
  quality: ResolvedGraphicsQuality;
  label: string;
  dpr: [number, number];
  shadows: boolean;
  shadowMapSize: 256 | 512 | 1024;
  maxDynamicLights: number;
  antialias: boolean;
  environmentIntensity: number;
};

export const GRAPHICS_PROFILES: Record<ResolvedGraphicsQuality, GraphicsProfile> = {
  high: {
    quality: 'high',
    label: 'High',
    dpr: [1, 1.5],
    shadows: false,
    shadowMapSize: 1024,
    maxDynamicLights: 8,
    antialias: true,
    environmentIntensity: 0.40,
  },
  balanced: {
    quality: 'balanced',
    label: 'Balanced',
    dpr: [1, 1],
    shadows: false,
    shadowMapSize: 512,
    maxDynamicLights: 4,
    antialias: true,
    environmentIntensity: 0.26,
  },
  compatibility: {
    quality: 'compatibility',
    label: 'Compatibility',
    dpr: [0.75, 1],
    shadows: false,
    shadowMapSize: 256,
    maxDynamicLights: 2,
    antialias: false,
    environmentIntensity: 0.16,
  },
};

type NavigatorWithGraphics = Navigator & {
  deviceMemory?: number;
  gpu?: {
    requestAdapter: (options?: { powerPreference?: 'low-power' | 'high-performance' }) => Promise<{
      info?: { description?: string; device?: string };
    } | null>;
  };
};

export const recommendGraphicsQuality = (
  capabilities: Pick<
    GraphicsCapabilities,
    'webgpuAvailable' | 'webgl2Available' | 'maxTextureSize' | 'logicalCores' | 'deviceMemoryGb'
  >,
): ResolvedGraphicsQuality => {
  if ((!capabilities.webgpuAvailable && !capabilities.webgl2Available) || capabilities.maxTextureSize < 8192) {
    return 'compatibility';
  }

  if (
    capabilities.logicalCores >= 16 &&
    capabilities.deviceMemoryGb !== undefined &&
    capabilities.deviceMemoryGb >= 16
  ) {
    return 'high';
  }

  return 'balanced';
};

export const resolveGraphicsQuality = (
  requested: GraphicsQuality,
  capabilities: GraphicsCapabilities,
): ResolvedGraphicsQuality => {
  if (!capabilities.webgpuAvailable && !capabilities.webgl2Available) return 'compatibility';
  if (requested === 'auto') return recommendGraphicsQuality(capabilities);
  return requested;
};

export const detectGraphicsCapabilities = async (): Promise<GraphicsCapabilities> => {
  const browserNavigator = navigator as NavigatorWithGraphics;
  const canvas = document.createElement('canvas');
  const webgl2 = canvas.getContext('webgl2', { failIfMajorPerformanceCaveat: true });
  const webgl = webgl2 || canvas.getContext('webgl', { failIfMajorPerformanceCaveat: true });
  const maxTextureSize = webgl
    ? Number(webgl.getParameter(webgl.MAX_TEXTURE_SIZE) || 0)
    : 0;
  const webgl2Available = Boolean(webgl2);
  const webglAvailable = Boolean(webgl);

  // Capability probing must not consume one of the browser's limited live
  // WebGL contexts; the actual R3F renderer is created immediately afterward.
  webgl?.getExtension('WEBGL_lose_context')?.loseContext();

  let webgpuAvailable = false;
  let webgpuAdapterName: string | undefined;
  if (browserNavigator.gpu) {
    try {
      const adapter = await browserNavigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      webgpuAvailable = Boolean(adapter);
      webgpuAdapterName = adapter?.info?.description || adapter?.info?.device || undefined;
    } catch {
      // Experimental browser implementations can expose navigator.gpu while
      // adapter creation still fails. WebGL remains the reliable fallback.
    }
  }

  return {
    webgpuAvailable,
    webgpuAdapterName,
    webgl2Available,
    webglAvailable,
    maxTextureSize,
    logicalCores: browserNavigator.hardwareConcurrency || 4,
    deviceMemoryGb: browserNavigator.deviceMemory,
  };
};

export type StoredRenderBaseline = {
  version: 1;
  capturedAt: string;
  jobId: string;
  requestedQuality: GraphicsQuality;
  resolvedQuality: ResolvedGraphicsQuality;
  capabilities: GraphicsCapabilities;
  metrics: RenderMetrics;
};

export const baselineStorageKey = (jobId: string, quality: ResolvedGraphicsQuality) =>
  `one-beyond:render-baseline:v1:${jobId}:${quality}`;

export const saveRenderBaseline = (baseline: StoredRenderBaseline) => {
  localStorage.setItem(
    baselineStorageKey(baseline.jobId, baseline.resolvedQuality),
    JSON.stringify(baseline),
  );
};
