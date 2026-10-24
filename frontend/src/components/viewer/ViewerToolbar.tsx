import React from 'react';
import {
  Sun,
  Moon,
  Sparkles,
  Grid,
  Camera,
  Maximize2,
  Minimize2,
  RotateCcw,
  ArrowLeft,
  Sliders,
  Gauge,
} from 'lucide-react';
import { GraphicsQuality, LightingMode, ResolvedGraphicsQuality } from '../../types';

interface ViewerToolbarProps {
  lightingMode: LightingMode;
  onSelectLighting: (mode: LightingMode) => void;
  showPointCloud: boolean;
  onTogglePointCloud: () => void;
  onResetCamera: () => void;
  onExit: () => void;
  graphicsQuality: GraphicsQuality;
  resolvedGraphicsQuality: ResolvedGraphicsQuality;
  onSelectGraphicsQuality: (quality: GraphicsQuality) => void;
  onCaptureScreenshot?: () => void;
  supportsPointCloud?: boolean;
  supportsWireframe?: boolean;
}

export const ViewerToolbar: React.FC<ViewerToolbarProps> = ({
  lightingMode,
  onSelectLighting,
  showPointCloud,
  onTogglePointCloud,
  onResetCamera,
  onExit,
  graphicsQuality,
  resolvedGraphicsQuality,
  onSelectGraphicsQuality,
  onCaptureScreenshot,
  supportsPointCloud = true,
  supportsWireframe = true,
}) => {
  const [isFullscreen, setIsFullscreen] = React.useState(false);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
        setIsFullscreen(false);
      }
    }
  };

  const handleCaptureScreenshot = () => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const dataUrl = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `one-beyond-space-${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    }
  };

  return (
    <div className="fixed top-4 right-4 z-40 flex items-center gap-2">
      {/* Exit Button */}
      <button
        onClick={onExit}
        className="glass-panel p-2.5 rounded-full hover:bg-white/10 text-gray-300 hover:text-white transition-all flex items-center gap-1.5 text-xs font-medium border border-white/10"
        title="Exit to Dashboard"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="hidden sm:inline">Exit</span>
      </button>

      {/* Lighting Mode Selector */}
      <div className="glass-panel p-1 rounded-full flex items-center gap-1 border border-white/10">
        <button
          onClick={() => onSelectLighting('cinematic')}
          className={`p-2 rounded-full transition-all ${
            lightingMode === 'cinematic'
              ? 'bg-cyan-500/20 text-cyan-300 shadow-sm shadow-cyan-500/30'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Cinematic Lighting"
        >
          <Moon className="w-4 h-4" />
        </button>
        <button
          onClick={() => onSelectLighting('studio')}
          className={`p-2 rounded-full transition-all ${
            lightingMode === 'studio'
              ? 'bg-cyan-500/20 text-cyan-300 shadow-sm shadow-cyan-500/30'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Studio Clean Light"
        >
          <Sparkles className="w-4 h-4" />
        </button>
        <button
          onClick={() => onSelectLighting('daylight')}
          className={`p-2 rounded-full transition-all ${
            lightingMode === 'daylight'
              ? 'bg-cyan-500/20 text-cyan-300 shadow-sm shadow-cyan-500/30'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Daylight Simulation"
        >
          <Sun className="w-4 h-4" />
        </button>
        <button
          onClick={() => onSelectLighting('wireframe')}
          disabled={!supportsWireframe}
          className={`p-2 rounded-full transition-all ${
            lightingMode === 'wireframe'
              ? 'bg-cyan-500/20 text-cyan-300 shadow-sm shadow-cyan-500/30'
              : 'text-gray-400 hover:text-white'
          }`}
          title={supportsWireframe ? 'Wireframe Spatial Mode' : 'Wireframe is available in the Three.js viewer'}
        >
          <Grid className="w-4 h-4" />
        </button>
      </div>

      {/* Renderer Quality Profile */}
      <label
        className="glass-panel h-10 px-3 rounded-full border border-white/10 flex items-center gap-2 text-gray-300"
        title={`Renderer quality: ${resolvedGraphicsQuality}`}
      >
        <Gauge className="w-4 h-4 text-cyan-400 shrink-0" />
        <select
          value={graphicsQuality}
          onChange={(event) => onSelectGraphicsQuality(event.target.value as GraphicsQuality)}
          onClick={(event) => event.stopPropagation()}
          className="bg-transparent text-[11px] font-mono uppercase outline-none cursor-pointer text-gray-200"
          aria-label="Graphics quality"
        >
          <option className="bg-gray-950" value="auto">Auto · {resolvedGraphicsQuality}</option>
          <option className="bg-gray-950" value="high">High</option>
          <option className="bg-gray-950" value="balanced">Balanced</option>
          <option className="bg-gray-950" value="compatibility">Compatibility</option>
        </select>
      </label>

      {/* Point Cloud Toggle */}
      <button
        onClick={onTogglePointCloud}
        disabled={!supportsPointCloud}
        className={`glass-panel p-2.5 rounded-full border transition-all ${
          showPointCloud
            ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
            : 'border-white/10 text-gray-400 hover:text-white'
        }`}
        title={supportsPointCloud ? 'Toggle 3D Feature Point Cloud' : 'Point cloud is available in the Three.js viewer'}
      >
        <Sliders className="w-4 h-4" />
      </button>

      {/* Reset Camera */}
      <button
        onClick={onResetCamera}
        className="glass-panel p-2.5 rounded-full hover:bg-white/10 text-gray-400 hover:text-white transition-all border border-white/10"
        title="Reset Camera Position"
      >
        <RotateCcw className="w-4 h-4" />
      </button>

      {/* Screenshot Capture */}
      <button
        onClick={onCaptureScreenshot || handleCaptureScreenshot}
        className="glass-panel p-2.5 rounded-full hover:bg-white/10 text-gray-400 hover:text-white transition-all border border-white/10"
        title="Capture Spatial Screenshot"
      >
        <Camera className="w-4 h-4" />
      </button>

      {/* Fullscreen Toggle */}
      <button
        onClick={toggleFullscreen}
        className="glass-panel p-2.5 rounded-full hover:bg-white/10 text-gray-400 hover:text-white transition-all border border-white/10"
        title="Toggle Fullscreen"
      >
        {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
      </button>
    </div>
  );
};
