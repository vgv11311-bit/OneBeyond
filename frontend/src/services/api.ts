import { SpaceJob, ReconstructionEngine } from '../types';

const API_BASE = '/api';
let reconnectRequest: Promise<void>|null=null;

async function boundedJson(url:string,options:RequestInit,timeout:number,message:string){
  const controller=new AbortController();
  let timer:ReturnType<typeof setTimeout>;
  const deadline=new Promise<never>((_,reject)=>{
    timer=setTimeout(()=>{controller.abort();reject(new Error(message));},timeout);
  });
  try {
    return await Promise.race([
      fetch(url,{...options,signal:controller.signal}).then(async response=>({response,data:await response.json()})),deadline,
    ]);
  } finally {clearTimeout(timer!);}
}

export function parseSpacesResponse(data: unknown): SpaceJob[] {
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Saved spaces response was malformed.');
  const payload = data as { spaces?: unknown; count?: unknown };
  if (!Array.isArray(payload.spaces) || !Number.isInteger(payload.count) || payload.count !== payload.spaces.length) {
    throw new Error('Saved spaces response was malformed.');
  }
  const ids = new Set<string>();
  for (const space of payload.spaces) {
    if (!space || typeof space !== 'object' || typeof space.job_id !== 'string' || !space.job_id ||
        (space.status !== 'completed' && space.status !== 'review_required') || ids.has(space.job_id)) {
      throw new Error('Saved spaces response was malformed.');
    }
    ids.add(space.job_id);
  }
  return payload.spaces as SpaceJob[];
}

export const api = {
  async reconnectResplat(): Promise<void> {
    if(!reconnectRequest)reconnectRequest=(async()=>{
      const {response,data}=await boundedJson(`${API_BASE}/reconstruction/resplat/connect`,{method:'POST',cache:'no-store'},45000,
        'GPU connection check timed out. The check has stopped; please reconnect again.');
      if(!response.ok||data?.can_run!==true)throw new Error(data?.availability_message||'Could not reconnect the running GPU.');
    })().finally(()=>{reconnectRequest=null;});
    return reconnectRequest;
  },

  async getHealth(): Promise<{ status: string; engines: ReconstructionEngine[] }> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch API health');
    return res.json();
  },

  async getEngines(): Promise<ReconstructionEngine[]> {
    const {response,data}=await boundedJson(`${API_BASE}/engines`,{cache:'no-store'},5000,'Runtime status refresh timed out. Please retry.');
    if(!response.ok)throw new Error('Failed to fetch engines');
    if(!data||!Array.isArray(data.engines))throw new Error('Runtime status response was malformed.');
    return data.engines;
  },

  async uploadMedia(files: File[], spaceName?: string): Promise<{ job_id: string; input_type: 'photos' | 'video'; capture_warnings: string[] }> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    if (spaceName) {
      formData.append('space_name', spaceName);
    }

    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Upload failed' }));
      throw new Error(err.error || 'Upload failed');
    }

    return res.json();
  },

  async startReconstruction(jobId: string, engineId: string = 'resplat_experimental', options: Record<string, unknown> = {}, attemptId?: string): Promise<void> {
    const res = await fetch(`${API_BASE}/reconstruct`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, engine_id: engineId, options, attempt_id: attemptId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Failed to start reconstruction' }));
      throw new Error(err.error || 'Failed to start reconstruction');
    }
  },

  async cancelReconstruction(job: Pick<SpaceJob, 'job_id' | 'attempt_id'>): Promise<void> {
    const res = await fetch(`${API_BASE}/reconstruct/cancel/${job.job_id}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attempt_id: job.attempt_id }), keepalive: true,
    });
    if (!res.ok) throw new Error('Could not cancel reconstruction. Please retry cancellation.');
  },

  async getJobStatus(jobId: string): Promise<SpaceJob> {
    const {response,data}=await boundedJson(`${API_BASE}/reconstruct/status/${jobId}`,{cache:'no-store'},8000,
      'Job status timed out. Retrying the connection; your last confirmed progress is retained.');
    if(!response.ok)throw new Error(`Failed to fetch job ${jobId} status`);
    if(!data||data.job_id!==jobId||!['uploaded','processing','completed','review_required','failed','cancelled'].includes(data.status))
      throw new Error('Job status response was malformed; last confirmed progress is retained.');
    return data;
  },

  async listSpaces(): Promise<SpaceJob[]> {
    const res = await fetch(`${API_BASE}/spaces`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Could not load saved spaces (HTTP ${res.status}).`);
    return parseSpacesResponse(await res.json());
  },

  async generateSampleSpace(): Promise<SpaceJob> {
    const res = await fetch(`${API_BASE}/sample/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error('Failed to generate sample space');
    return res.json();
  },

  getModelUrl(jobId: string): string {
    return `${API_BASE}/spaces/${jobId}/model.glb`;
  },
};
