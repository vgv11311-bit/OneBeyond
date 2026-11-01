import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { Navbar } from './components/Navbar';
import { LandingHero } from './components/LandingHero';
import { UploadModal } from './components/UploadModal';
import { ProcessingView } from './components/ProcessingView';
const SpatialViewer = lazy(() => import('./components/viewer/RoomViewer').then(module => ({ default: module.RoomViewer })));
import { api } from './services/api';
import { resumeActiveReconstruction } from './services/reconstructionSession';
import { SavedSpacesController, SavedSpacesState, readSavedSpacesCache } from './services/savedSpaces';
import { ViewerErrorBoundary } from './components/viewer/ViewerErrorBoundary';
import { SpaceJob, ReconstructionEngine, AppView } from './types';
import { currentSiteRoute } from './siteRoutes';

export const App: React.FC = () => {
  const [view, setView] = useState<AppView>('landing');
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isSubmittingUpload, setIsSubmittingUpload] = useState(false);
  const [currentJob, setCurrentJob] = useState<SpaceJob | null>(null);
  const [pollWarning,setPollWarning]=useState('');
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

  const [restoreWarning,setRestoreWarning]=useState('');
  const restoreGeneration=useRef(0);
  const restoreSavedJob=async()=>{
    const generation=++restoreGeneration.current;
    try{
      const job=await resumeActiveReconstruction(sessionStorage,id=>api.getJobStatus(id));
      if(generation!==restoreGeneration.current)return;
      setRestoreWarning('');
      if(job){setCurrentJob(job);setView('processing');}
    }catch{
      if(generation===restoreGeneration.current)setRestoreWarning('Could not reconnect to the saved reconstruction. Your job was not cancelled or restarted.');
    }
  };
  useEffect(()=>{void restoreSavedJob();return ()=>{restoreGeneration.current++;};},[]);
  useEffect(()=>{
    if(!currentJob?.attempt_id)return;
    if(currentJob.status==='processing'){
      sessionStorage.setItem('activeReconstruction',JSON.stringify({job_id:currentJob.job_id,attempt_id:currentJob.attempt_id}));
    }else sessionStorage.removeItem('activeReconstruction');
  },[currentJob?.job_id,currentJob?.attempt_id,currentJob?.status]);

  const handleCancel = async () => {
    ++restoreGeneration.current;setRestoreWarning('');
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
    setPollWarning('');
    let pending = false;
    if (view === 'processing' && currentJob && !['completed','review_required','failed','cancelled'].includes(currentJob.status)) {
      pollIntervalRef.current = window.setInterval(async () => {
        if (pending) return;
        pending = true;
        try {
          const updated = await api.getJobStatus(currentJob.job_id);
          if (generation !== pollGeneration.current) return;
          setPollWarning('');
          setCurrentJob(previous=>previous?.job_id===updated.job_id && previous.attempt_id===updated.attempt_id && updated.status==='processing'
            ? {...updated,progress:Math.max(previous.progress||0,updated.progress||0)} : updated);
          if (['completed','review_required','failed','cancelled'].includes(updated.status)) {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            // Refresh spaces list
            refreshSpaces();
          }
        } catch (e) {
          if(generation===pollGeneration.current)setPollWarning(e instanceof Error?e.message:'Connection to the job was lost. Retrying automatically.');
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
    ++restoreGeneration.current;setRestoreWarning('');
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
    ++restoreGeneration.current;setRestoreWarning('');
    if (!currentJob || currentJob.status !== 'failed' || isRetrying) return;
    setIsRetrying(true);
    ++pollGeneration.current;
    try {
      await api.startReconstruction(currentJob.job_id, currentJob.engine_id || 'resplat_experimental',
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
    ++restoreGeneration.current;setRestoreWarning('');
    try {
      if (currentJob?.status === 'processing') await api.cancelReconstruction(currentJob);
      const sampleJob = await api.generateSampleSpace();
      setCurrentJob(sampleJob);
      setView('viewer');
      refreshSpaces();
    } catch (e: any) {
      console.error('Failed to launch sample space:', e);
      alert('The sample space could not be opened. Please try again.');
    }
  };

  const handleSelectSpace = async (space: SpaceJob) => {
    ++restoreGeneration.current;setRestoreWarning('');
    const generation = ++selectionGeneration.current;
    setOpeningSpace(true);
    setOpenSpaceError('');
    try {
      // Gallery cards intentionally omit camera, lighting and object metadata.
      // Always hydrate the saved scene before mounting the viewer.
      const fullSpace = await api.getJobStatus(space.job_id);
      if (generation !== selectionGeneration.current) return;
      const savedReview = fullSpace.status === 'review_required' && ['resplat_experimental'].includes(fullSpace.engine_id || '') && fullSpace.gaussian?.saved_to_recents === true && fullSpace.gaussian?.requires_review === true && fullSpace.gaussian?.quality_accepted === false;
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
          connectionWarning={pollWarning}
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

      {restoreWarning && <div role="alert" className="fixed bottom-5 left-5 right-5 z-50 rounded-xl border border-amber-500/40 bg-[#1b1815] p-4 text-sm text-amber-100">{restoreWarning} <button className="underline" onClick={()=>void restoreSavedJob()}>Reconnect saved reconstruction</button></div>}

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSubmit={handleUploadSubmit}
        engines={engines}
        onRefreshEngines={async(engineId)=>{
          let connectionError:unknown;
          try{if(engineId==='resplat_experimental')await api.reconnectResplat();}
          catch(error){connectionError=error;}
          const latest=await api.getEngines();
          setEngines(latest);
          if(connectionError&&latest.find(engine=>engine.id===engineId)?.can_run!==true)throw connectionError;
        }}
        isSubmitting={isSubmittingUpload}
      />
    </div>
  );
};

export default App;
