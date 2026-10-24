import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { Navbar } from './components/Navbar';
import { LandingHero } from './components/LandingHero';
import { UploadModal } from './components/UploadModal';
import { ProcessingView } from './components/ProcessingView';
const SpatialViewer = lazy(() => import('./components/viewer/RoomViewer').then(module => ({ default: module.RoomViewer })));
import { api } from './services/api';
import { SavedSpacesController, SavedSpacesState, readSavedSpacesCache } from './services/savedSpaces';
import { ViewerErrorBoundary } from './components/viewer/ViewerErrorBoundary';
import { SpaceJob, ReconstructionEngine, AppView } from './types';
import { currentSiteRoute } from './siteRoutes';

export const App: React.FC = () => {
  const [view, setView] = useState<AppView>('landing');
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isSubmittingUpload, setIsSubmittingUpload] = useState(false);
  const [currentJob, setCurrentJob] = useState<SpaceJob | null>(null);
  const [engines, setEngines] = useState<ReconstructionEngine[]>([]);
  const [spacesState, setSpacesState] = useState<SavedSpacesState>(() => ({
    spaces: readSavedSpacesCache(sessionStorage), status: 'loading', error: '',
  }));
  const spacesControllerRef = useRef<SavedSpacesController | null>(null);
  if (!spacesControllerRef.current) {
    spacesControllerRef.current = new SavedSpacesController(() => api.listSpaces(), setSpacesState, sessionStorage);
  }
  const refreshSpaces = () => { void spacesControllerRef.current?.refresh(); };
  const pollIntervalRef = useRef<number | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);
  const pollGeneration = useRef(0);
  const selectionGeneration = useRef(0);
  const [openingSpace, setOpeningSpace] = useState(false);
  const [openSpaceError, setOpenSpaceError] = useState('');

  useEffect(() => {
    // Retry cancellation after reload if the unload beacon did not arrive.
    const abandoned = sessionStorage.getItem('activeReconstruction');
    if (abandoned) {
      try { api.cancelReconstruction(JSON.parse(abandoned)).then(() => sessionStorage.removeItem('activeReconstruction')).catch(console.error); }
      catch { sessionStorage.removeItem('activeReconstruction'); }
    }
  }, []);

  useEffect(() => {
    if (currentJob?.status !== 'processing' || !currentJob.attempt_id) return;
    const active = { job_id: currentJob.job_id, attempt_id: currentJob.attempt_id };
    sessionStorage.setItem('activeReconstruction', JSON.stringify(active));
    const abandon = () => {
      navigator.sendBeacon(`/api/reconstruct/cancel/${active.job_id}`,
        new Blob([JSON.stringify(active)], { type: 'application/json' }));
    };
    window.addEventListener('pagehide', abandon);
    window.addEventListener('beforeunload', abandon);
    return () => {
      window.removeEventListener('pagehide', abandon);
      window.removeEventListener('beforeunload', abandon);
      sessionStorage.removeItem('activeReconstruction');
    };
  }, [currentJob?.job_id, currentJob?.attempt_id, currentJob?.status]);

  const handleCancel = async () => {
    if (currentJob?.status === 'processing') {
      try { await api.cancelReconstruction(currentJob); }
      catch (e: any) { alert(e.message); return; }
      ++pollGeneration.current;
      setCurrentJob(job => job ? { ...job, status: 'cancelled' } : job);
    }
    setView('landing');
  };

  // Initial data loading
  useEffect(() => {
    if (view !== 'landing') return;
      let active = true;
      // Provider readiness must not delay access to already saved rooms.
      api.getEngines().then((engs) => { if (active) setEngines(engs); }).catch(console.error);
      refreshSpaces();
      const refreshOnSpacesNavigation = () => {
        if (currentSiteRoute() === 'recent') refreshSpaces();
      };
      window.addEventListener('popstate', refreshOnSpacesNavigation);
      return () => {
        active = false;
        window.removeEventListener('popstate', refreshOnSpacesNavigation);
        spacesControllerRef.current?.invalidate();
      };
  }, [view]);

  // Polling for active job status
  useEffect(() => {
    const generation = ++pollGeneration.current;
    let pending = false;
    if (view === 'processing' && currentJob && !['completed','review_required','failed','cancelled'].includes(currentJob.status)) {
      pollIntervalRef.current = window.setInterval(async () => {
        if (pending) return;
        pending = true;
        try {
          const updated = await api.getJobStatus(currentJob.job_id);
          if (generation !== pollGeneration.current) return;
          setCurrentJob(updated);
          if (['completed','review_required','failed','cancelled'].includes(updated.status)) {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            // Refresh spaces list
            refreshSpaces();
          }
        } catch (e) {
          console.error('Polling error:', e);
        } finally {
          pending = false;
        }
      }, 750);
    }

    return () => {
      ++pollGeneration.current;
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [view, currentJob?.job_id, currentJob?.status]);

  const handleUploadSubmit = async (
    files: File[],
    spaceName: string,
    engineId: string,
    options: { knownRoomWidthM?: number; photographic?: boolean },
  ) => {
    setIsSubmittingUpload(true);
    try {
      if (currentJob?.status === 'processing') await api.cancelReconstruction(currentJob);
      const uploadRes = await api.uploadMedia(files, spaceName);
      
      const initialJobState: SpaceJob = {
        job_id: uploadRes.job_id,
        attempt_id: crypto.randomUUID(),
        name: spaceName,
        created_at: new Date().toISOString(),
        status: 'processing',
        input_type: uploadRes.input_type,
        files_count: files.length,
        progress: 5.0,
        stage: 'Initializing reconstruction...',
        logs: [`Uploaded ${files.length} files. Starting ${engineId} engine...`, ...(uploadRes.capture_warnings || [])],
        engine_id: engineId,
        reconstruction_options: options,
      };

      setCurrentJob(initialJobState);
      setIsUploadOpen(false);
      setView('processing');

      // Trigger background reconstruction
      await api.startReconstruction(uploadRes.job_id, engineId, options, initialJobState.attempt_id);
    } catch (e: any) {
      setCurrentJob(job => job ? { ...job, status: 'failed', error: e.message || 'Could not start reconstruction' } : job);
      alert(e.message || 'Upload failed');
    } finally {
      setIsSubmittingUpload(false);
    }
  };

  const handleRetry = async () => {
    if (!currentJob || currentJob.status !== 'failed' || isRetrying) return;
    setIsRetrying(true);
    ++pollGeneration.current;
    try {
      await api.startReconstruction(currentJob.job_id, currentJob.engine_id || 'photographic',
        currentJob.reconstruction_options || {}, crypto.randomUUID());
      // Read the acknowledged server state; a fast failure or completion must
      // not be overwritten with a fabricated processing state.
      const updated = await api.getJobStatus(currentJob.job_id);
      setCurrentJob(updated);
    } catch (e: any) {
      setCurrentJob(job => job ? { ...job, error: e.message || 'Retry could not start' } : job);
    } finally {
      setIsRetrying(false);
    }
  };

  const handleExploreSample = async () => {
    try {
      if (currentJob?.status === 'processing') await api.cancelReconstruction(currentJob);
      const sampleJob = await api.generateSampleSpace();
      setCurrentJob(sampleJob);
      setView('viewer');
      refreshSpaces();
    } catch (e: any) {
      alert('Failed to launch sample studio: ' + e.message);
    }
  };

  const handleSelectSpace = async (space: SpaceJob) => {
    const generation = ++selectionGeneration.current;
    setOpeningSpace(true);
    setOpenSpaceError('');
    try {
      // Gallery cards intentionally omit camera, lighting and object metadata.
      // Always hydrate the saved scene before mounting the viewer.
      const fullSpace = await api.getJobStatus(space.job_id);
      if (generation !== selectionGeneration.current) return;
      const savedReview = fullSpace.status === 'review_required' && fullSpace.engine_id === 'resplat_experimental' && fullSpace.gaussian?.saved_to_recents === true && fullSpace.gaussian?.requires_review === true;
      if (fullSpace.status !== 'completed' && !savedReview) throw new Error('This saved room is not ready to open.');
      setCurrentJob(fullSpace);
      setView('viewer');
    } catch (error) {
      if (generation === selectionGeneration.current) setOpenSpaceError(error instanceof Error ? error.message : 'Could not open saved room. Please retry.');
    } finally {
      if (generation === selectionGeneration.current) setOpeningSpace(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#07070a] text-white flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Show Navbar on non-viewer screens */}
      {view !== 'viewer' && (
        <Navbar
          onOpenUpload={() => setIsUploadOpen(true)}
          onSelectSample={handleExploreSample}
          onHomeClick={handleCancel}
          activeView={view}
        />
      )}

      {/* Main View Flow */}
      {view === 'landing' && openingSpace && <p role="status" className="px-8 py-2 text-cyan-300">Opening saved room…</p>}
      {view === 'landing' && openSpaceError && <p role="alert" className="px-8 py-2 text-red-300">{openSpaceError}</p>}
      {view === 'landing' && (
        <LandingHero
          onOpenUpload={() => setIsUploadOpen(true)}
          onExploreSample={handleExploreSample}
          onSelectSpace={handleSelectSpace}
          recentSpaces={spacesState.spaces}
          spacesStatus={spacesState.status}
          spacesError={spacesState.error}
          onRetrySpaces={refreshSpaces}
          isUploadOpen={isUploadOpen}
        />
      )}

      {view === 'processing' && currentJob && (
        <ProcessingView
          job={currentJob}
          onEnterViewer={() => setView('viewer')}
          onCancel={handleCancel}
          onRetry={handleRetry}
          isRetrying={isRetrying}
        />
      )}

      {view === 'viewer' && currentJob && (
        <ViewerErrorBoundary key={currentJob.job_id} onRetry={() => window.location.reload()}>
        <Suspense fallback={<div role="status" className="p-8 text-white">Loading 3D viewer…</div>}>
        <SpatialViewer
          job={currentJob}
          onExit={() => setView('landing')}
        />
        </Suspense>
        </ViewerErrorBoundary>
      )}

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSubmit={handleUploadSubmit}
        engines={engines}
        onRefreshEngines={async()=>{setEngines(await api.getEngines());}}
        isSubmitting={isSubmittingUpload}
      />
    </div>
  );
};

export default App;
