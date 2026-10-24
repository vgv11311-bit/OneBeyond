import { describe, expect, it } from 'vitest';
import {
  GRAPHICS_PROFILES,
  baselineStorageKey,
  recommendGraphicsQuality,
  resolveGraphicsQuality,
} from './graphicsProfiles';
import { GraphicsCapabilities } from '../../types';

const capableDesktop: GraphicsCapabilities = {
  webgpuAvailable: true,
  webgl2Available: true,
  webglAvailable: true,
  maxTextureSize: 16384,
  logicalCores: 16,
  deviceMemoryGb: 16,
};

describe('graphics quality selection', () => {
  it('selects high for a capable WebGL2 desktop', () => {
    expect(recommendGraphicsQuality(capableDesktop)).toBe('high');
  });

  it('selects compatibility when WebGL2 is unavailable', () => {
    const unsupported = { ...capableDesktop, webgpuAvailable: false, webgl2Available: false };
    expect(recommendGraphicsQuality(unsupported)).toBe('compatibility');
    expect(resolveGraphicsQuality('high', unsupported)).toBe('compatibility');
  });

  it('allows WebGPU-first devices without WebGL2', () => {
    expect(resolveGraphicsQuality('balanced', { ...capableDesktop, webgl2Available: false })).toBe('balanced');
  });

  it('selects balanced for moderate hardware', () => {
    expect(recommendGraphicsQuality({
      ...capableDesktop,
      logicalCores: 4,
      deviceMemoryGb: 4,
    })).toBe('balanced');
  });

  it('prefers balanced on typical laptops to protect walk performance', () => {
    expect(recommendGraphicsQuality({
      ...capableDesktop,
      logicalCores: 12,
    })).toBe('balanced');
  });

  it('keeps explicit quality choices on WebGL2', () => {
    expect(resolveGraphicsQuality('balanced', capableDesktop)).toBe('balanced');
    expect(resolveGraphicsQuality('compatibility', capableDesktop)).toBe('compatibility');
  });
});

describe('graphics profile invariants', () => {
  it('keeps the temporary shadow disable active at every quality', () => {
    for (const profile of Object.values(GRAPHICS_PROFILES)) {
      expect(profile.shadows).toBe(false);
    }
  });
  it('reduces expensive work progressively', () => {
    expect(GRAPHICS_PROFILES.high.shadowMapSize).toBeGreaterThan(GRAPHICS_PROFILES.balanced.shadowMapSize);
    expect(GRAPHICS_PROFILES.balanced.shadowMapSize).toBeGreaterThan(GRAPHICS_PROFILES.compatibility.shadowMapSize);
    expect(GRAPHICS_PROFILES.compatibility.shadows).toBe(false);
    expect(GRAPHICS_PROFILES.high.maxDynamicLights).toBeGreaterThan(GRAPHICS_PROFILES.compatibility.maxDynamicLights);
  });

  it('creates stable job and quality scoped baseline keys', () => {
    expect(baselineStorageKey('room-123', 'balanced')).toBe(
      'one-beyond:render-baseline:v1:room-123:balanced',
    );
  });
});
