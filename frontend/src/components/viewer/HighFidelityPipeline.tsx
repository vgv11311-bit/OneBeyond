import React, { useEffect, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE_WEBGPU from 'three/webgpu';
import { pass } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { ResolvedGraphicsQuality } from '../../types';

interface HighFidelityPipelineProps {
  quality: ResolvedGraphicsQuality;
  environmentIntensity: number;
  photoMode?: boolean;
}

/**
 * Three's native node render pipeline. The same TSL graph compiles to WGSL on
 * WebGPU and GLSL on the universal renderer's WebGL2 fallback.
 */
export const HighFidelityPipeline: React.FC<HighFidelityPipelineProps> = ({
  quality,
  environmentIntensity,
  photoMode = false,
}) => {
  const { gl, scene, camera } = useThree();
  const renderer = gl as unknown as THREE_WEBGPU.WebGPURenderer;

  const pipelineState = useMemo(() => {
    const renderPipeline = new THREE_WEBGPU.RenderPipeline(renderer);
    const scenePass = pass(scene, camera);
    const sceneColor = scenePass.getTextureNode('output');
    const disposables: Array<{ dispose: () => void }> = [renderPipeline, scenePass];
    let outputNode: any = sceneColor;

    if (quality === 'high' && !photoMode) {
      // Subtle selective-feeling bloom: the high threshold protects walls and
      // furniture colors while allowing lamps and bright window highlights to glow.
      const bloomPass = bloom(sceneColor, 0.1, 0.18, 0.92);
      outputNode = sceneColor.add(bloomPass);
      disposables.push(bloomPass);
    }

    if (quality !== 'compatibility') {
      const fxaaPass = fxaa(outputNode);
      renderPipeline.outputNode = fxaaPass;
      disposables.push(fxaaPass);
    } else {
      renderPipeline.outputNode = outputNode;
    }

    return { renderPipeline, disposables };
  }, [renderer, scene, camera, quality, photoMode]);

  useEffect(() => {
    const previousEnvironment = scene.environment;
    const previousIntensity = scene.environmentIntensity;
    const roomEnvironment = new RoomEnvironment();
    const pmremGenerator = new THREE_WEBGPU.PMREMGenerator(renderer as never);
    const environmentTarget = pmremGenerator.fromScene(roomEnvironment, 0.04);
    scene.environment = environmentTarget.texture;
    scene.environmentIntensity = environmentIntensity;

    return () => {
      scene.environment = previousEnvironment;
      scene.environmentIntensity = previousIntensity;
      environmentTarget.dispose();
      pmremGenerator.dispose();
      roomEnvironment.dispose();
    };
  }, [renderer, scene, environmentIntensity]);

  useEffect(() => () => {
    pipelineState.disposables.forEach((item) => item.dispose());
  }, [pipelineState]);

  useEffect(() => {
    const previous = renderer.info.autoReset;
    renderer.info.autoReset = false;
    return () => { renderer.info.autoReset = previous; };
  }, [renderer]);

  // Positive priority replaces R3F's default render call with this HDR graph.
  useFrame(() => {
    renderer.info.reset();
    pipelineState.renderPipeline.render();
    // Snapshot after all scene/postprocessing passes, before the next reset.
    renderer.domElement.dataset.frameWork = JSON.stringify({
      drawCalls: renderer.info.render.drawCalls,
      triangles: renderer.info.render.triangles,
    });
  }, 1);

  return null;
};
