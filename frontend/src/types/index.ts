export type ReconstructionEngine = {
  id: string;
  name: string;
  description: string;
  version: string;
  is_default: boolean;
  experimental?: boolean;
  can_run?: boolean;
  availability_code?: string;
  availability_message?: string;
};

export type ReconstructionMetrics = {
  total_frames_processed: number;
  keypoints_detected: number;
  points_3d_triangulated: number;
  estimated_room_width: number;
  estimated_room_length: number;
  estimated_room_height: number;
  processing_duration_sec: number;
  engine_name: string;
  extra_details?: Record<string, any>;
};

export type JobStatus = 'uploaded' | 'processing' | 'completed' | 'review_required' | 'failed' | 'cancelled';

export type SpaceJob = {
  job_id: string;
  attempt_id?: string;
  name: string;
  created_at: string;
  status: JobStatus;
  input_type: 'photos' | 'video' | 'sample';
  files_count: number;
  progress: number;
  stage: string;
  logs: string[];
  glb_url?: string;
  gaussian?: {
    status: 'ready' | 'unavailable' | 'skipped';
    viewer_url?: string;
    ply_url?: string;
    config_url?: string;
    notice?: string;
    reason?: string;
    vertices?: number;
    mesh_url?: string;
    experimental?: boolean;
    requires_review?: boolean;
    quality_accepted?: boolean;
    saved_to_recents?: boolean;
  };
  metrics?: ReconstructionMetrics;
  error?: string;
  error_code?: string;
  retry_after_seconds?: number;
  engine_id?: string;
  reconstruction_options?: { knownRoomWidthM?: number; photographic?: boolean };
};

export type LightingMode = 'cinematic' | 'studio' | 'daylight' | 'wireframe';

export type GraphicsQuality = 'auto' | 'high' | 'balanced' | 'compatibility';
export type ResolvedGraphicsQuality = Exclude<GraphicsQuality, 'auto'>;

export type GraphicsCapabilities = {
  webgpuAvailable: boolean;
  webgpuAdapterName?: string;
  webgl2Available: boolean;
  webglAvailable: boolean;
  maxTextureSize: number;
  logicalCores: number;
  deviceMemoryGb?: number;
};

export type RenderMetrics = {
  backend: 'WebGPU' | 'WebGPU → WebGL2' | 'WebGL2' | 'WebGL1';
  fps: number;
  averageFrameMs: number;
  p95FrameMs: number;
  drawCalls: number;
  triangles: number;
  geometries: number;
  textures: number;
  renderWidth: number;
  renderHeight: number;
  modelLoadMs?: number;
  heapMemoryMb?: number;
};

export type InferredLighting = {
  daylight?: { version: number; direct_sun: boolean; reason: string; windows: Array<{
    center: [number,number,number]; size: [number,number,number]; wall: string;
    target?: [number,number,number]; color: [number,number,number];
    diffuse_intensity: number; direct_intensity: number;
  }> };
  analysis_warning?: string;
  projection_camera?: { intrinsics: number[][]; source_shape: [number, number]; height: number; position?: [number,number,number]; yaw_degrees?: number; rotation_matrix?: number[][] };
  brightness: number;
  contrast: number;
  warmth: number;
  key_image_position: [number, number];
  ambient_intensity: number;
  key_intensity: number;
  exposure: number;
  surface_colors?: {
    wall: [number, number, number];
    floor: [number, number, number];
    ceiling: [number, number, number];
  };
};

export type AppView = 'landing' | 'upload' | 'processing' | 'viewer';
