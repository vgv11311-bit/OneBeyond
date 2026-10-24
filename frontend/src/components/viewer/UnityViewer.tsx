import React,{useEffect,useId,useRef,useState} from 'react';
import { GraphicsQuality,LightingMode,ResolvedGraphicsQuality,SpaceJob } from '../../types';
import { ViewerToolbar } from './ViewerToolbar';
import { SpatialHUD } from './SpatialHUD';
import { loadUnityLoader,UnityEvent,UnityInstance,unityBuildConfig,unityPlayerLease,unitySceneRequest } from './unityBridge';

export const UnityViewer:React.FC<{job:SpaceJob;onExit:()=>void;onFallback:(reason:string)=>void}>=({job,onExit,onFallback})=>{
  // Unity's WebGL bridge addresses its canvas by a CSS ID selector.
  const canvasId=`unity-canvas-${useId().replace(/[^a-zA-Z0-9_-]/g,'')}`;
  const canvas=useRef<HTMLCanvasElement>(null), player=useRef<UnityInstance|undefined>(undefined);
  const [progress,setProgress]=useState(0),[loaded,setLoaded]=useState(false),[error,setError]=useState('');
  const [quality,setQuality]=useState<GraphicsQuality>(()=>{
    const saved=localStorage.getItem('one-beyond:graphics-quality');
    return saved==='high'||saved==='balanced'||saved==='compatibility'?saved:'auto';
  });
  const [resolved,setResolved]=useState<ResolvedGraphicsQuality>('balanced');
  const [lighting,setLighting]=useState<LightingMode>('cinematic');
  const [locked,setLocked]=useState(false),[clear,setClear]=useState(false),[fps,setFps]=useState(0);
  const heightTimer=useRef<ReturnType<typeof setTimeout>|undefined>(undefined);
  const movement=useRef({x:0,z:0,y:0});
  const currentQuality=useRef(quality),currentLighting=useRef(lighting);
  const send=(method:string,value='')=>player.current?.SendMessage('OneBeyondViewer',method,value);
  const lock=()=>{
    canvas.current?.focus();send('SetFocused','true');
    const result=canvas.current?.requestPointerLock();
    if(result&&'catch' in result) result.catch(()=>undefined);
  };
  useEffect(()=>{
    const element=canvas.current!;
    const controller=new AbortController();let disposed=false,instance:UnityInstance|undefined,startupDone=false;
    let quitting:Promise<void>|undefined;
    const quit=()=>{if(!quitting)quitting=(instance?instance.Quit():Promise.resolve()).catch(()=>undefined).finally(lease.release);return quitting;};
    const lease=unityPlayerLease();const session=crypto.randomUUID();let timeout:ReturnType<typeof setTimeout>;
    const fail=(message:string)=>{if(!disposed){clearTimeout(timeout);setError(message);}};
    let ready=false;
    const configure=()=>{
      if(!instance||!ready||disposed) return;
      instance.SendMessage('OneBeyondViewer','SetQuality',currentQuality.current);
      instance.SendMessage('OneBeyondViewer','LoadScene',JSON.stringify(unitySceneRequest(job,session,location.origin)));
    };
    const onEvent=(event:Event)=>{
      let detail:UnityEvent;try{detail=JSON.parse((event as CustomEvent<string>).detail);}catch{return;}
      if(disposed) return;
      if(detail.type==='ready'){ready=true;configure();return;}
      if(detail.type!=='quality'&&detail.session!==session) return;
      if(detail.type==='loaded'){clearTimeout(timeout);setLoaded(true);instance?.SendMessage('OneBeyondViewer','SetLighting',currentLighting.current);}
      if(detail.type==='error') fail(detail.message||'Room loading failed.');
      if(detail.type==='metrics') setFps(Math.round(detail.fps||0));
      if((detail.type==='quality'||detail.type==='metrics')&&['high','balanced','compatibility'].includes(detail.quality||'')) setResolved(detail.quality as ResolvedGraphicsQuality);
      if(detail.type==='screenshot'&&detail.message){const link=document.createElement('a');link.download=`one-beyond-${Date.now()}.png`;link.href=`data:image/png;base64,${detail.message}`;link.click();}
    };
    const focus=()=>instance?.SendMessage('OneBeyondViewer','SetFocused','true');
    const blur=()=>{instance?.SendMessage('OneBeyondViewer','SetFocused','false');instance?.SendMessage('OneBeyondViewer','SetMotion','{"x":0,"y":0,"z":0}');};
    const pointer=()=>setLocked(document.pointerLockElement===element);
    let previousTouch:{x:number;y:number}|undefined;
    const look=(x:number,y:number)=>instance?.SendMessage('OneBeyondViewer','Look',JSON.stringify({x,y,z:0}));
    const mouse=(event:MouseEvent)=>{if(document.pointerLockElement===element||event.buttons===1&&event.target===element)look(event.movementX,event.movementY);};
    const touchStart=(event:TouchEvent)=>{const t=event.touches[0];previousTouch=t?{x:t.clientX,y:t.clientY}:undefined;focus();};
    const touchMove=(event:TouchEvent)=>{const t=event.touches[0];if(t&&previousTouch){look(t.clientX-previousTouch.x,t.clientY-previousTouch.y);previousTouch={x:t.clientX,y:t.clientY};event.preventDefault();}};
    const touchEnd=()=>{previousTouch=undefined;};
    element.addEventListener('one-beyond-unity',onEvent);element.addEventListener('focus',focus);element.addEventListener('blur',blur);
    window.addEventListener('blur',blur);document.addEventListener('pointerlockchange',pointer);
    document.addEventListener('mousemove',mouse);element.addEventListener('touchstart',touchStart,{passive:true});
    element.addEventListener('touchmove',touchMove,{passive:false});element.addEventListener('touchend',touchEnd);
    element.addEventListener('touchcancel',touchEnd);
    timeout=setTimeout(()=>fail('Unity startup exceeded two minutes. Retry or use the existing viewer.'),120000);
    void (async()=>{
      try{
        await lease.wait;if(disposed) return;
        const response=await fetch('/unity/viewer.json',{signal:controller.signal,cache:'no-cache'});
        if(!response.ok||!response.headers.get('content-type')?.includes('json')) throw new Error('Unity web build is not published yet.');
        const config=unityBuildConfig(await response.json(),location.origin);
        await loadUnityLoader(config.loaderUrl);if(disposed) return;
        instance=await window.createUnityInstance!(element,{...config,companyName:'One Beyond',productName:'Spatial Viewer',productVersion:'1.0',
          matchWebGLToCanvasSize:true,devicePixelRatio:Math.min(devicePixelRatio,1.5),
          showBanner:(message:string,type:string)=>{if(type==='error')fail(message);},
        },value=>{if(!disposed)setProgress(Math.round(value*100));});
        if(disposed){await quit();return;}
        player.current=instance;configure();
      }catch(e){if(!disposed)fail(e instanceof Error?e.message:typeof e==='string'?e:'Unity could not start.');}
      finally{startupDone=true;if(disposed)await quit();}
    })();
    return ()=>{
      disposed=true;clearTimeout(timeout);controller.abort();player.current=undefined;
      clearTimeout(heightTimer.current);
      element.removeEventListener('one-beyond-unity',onEvent);element.removeEventListener('focus',focus);element.removeEventListener('blur',blur);
      window.removeEventListener('blur',blur);document.removeEventListener('pointerlockchange',pointer);
      document.removeEventListener('mousemove',mouse);element.removeEventListener('touchstart',touchStart);
      element.removeEventListener('touchmove',touchMove);element.removeEventListener('touchend',touchEnd);element.removeEventListener('touchcancel',touchEnd);
      if(document.pointerLockElement===element) document.exitPointerLock();
      // Startup can be pending: its finally block releases the lease after Quit.
      if(startupDone) void quit();
    };
  },[job]);
  useEffect(()=>{currentQuality.current=quality;localStorage.setItem('one-beyond:graphics-quality',quality);send('SetQuality',quality);},[quality]);
  useEffect(()=>{currentLighting.current=lighting;send('SetLighting',lighting);},[lighting]);
  return <div className="fixed inset-0 z-50 bg-[#07070a]">
    <canvas id={canvasId} ref={canvas} tabIndex={0} aria-label="Unity reconstructed room. Click to look; WASD to walk; Escape to release the mouse." onClick={lock} onContextMenu={e=>e.preventDefault()} className="h-full w-full outline-none touch-none"/>
    {!clear&&<><ViewerToolbar lightingMode={lighting} onSelectLighting={setLighting} showPointCloud={false} onTogglePointCloud={()=>{}}
      graphicsQuality={quality} resolvedGraphicsQuality={resolved} onSelectGraphicsQuality={setQuality} onResetCamera={()=>send('ResetCamera')}
      onExit={onExit} supportsPointCloud={false} supportsWireframe={false} onCaptureScreenshot={()=>send('Capture')}/>
      <SpatialHUD job={job} isPointerLocked={locked} onLockPointer={lock} graphicsQuality={quality} resolvedGraphicsQuality={resolved}
        baselineCaptured={false} onCaptureBaseline={()=>send('Capture')}
        onVirtualMove={(x,z)=>{movement.current={...movement.current,x,z};send('SetFocused','true');send('SetMotion',JSON.stringify(movement.current));}}
        onVirtualHeight={y=>{movement.current.y=y;send('SetFocused','true');send('SetMotion',JSON.stringify(movement.current));
          clearTimeout(heightTimer.current);heightTimer.current=setTimeout(()=>{movement.current.y=0;send('SetMotion',JSON.stringify(movement.current));},200);}}/>
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-xs text-slate-300">Unity URP · {resolved}{fps>0?` · ${fps} FPS`:''}</div>
    </>}
    <div className="fixed bottom-4 right-4 z-50 flex gap-2">
      <button className="rounded-full bg-black/80 px-4 py-2 text-xs text-white" onClick={()=>onFallback('Switched manually.')}>Use Three.js</button>
      <button className="rounded-full bg-black/80 px-4 py-2 text-xs text-white" onClick={()=>setClear(!clear)}>{clear?'Show controls':'Clear view'}</button>
    </div>
    {!loaded&&!error&&<div role="status" className="absolute inset-0 z-20 grid place-content-center bg-black/80 text-center text-cyan-200 pointer-events-none">Loading Unity viewer… {progress}%<p className="mt-2 text-xs text-slate-400">The first visit downloads the player. No installation required.</p></div>}
    {error&&<div role="alert" className="absolute inset-0 z-50 grid place-content-center gap-4 bg-black/90 p-8 text-center text-white"><p>{error}</p><button className="rounded bg-cyan-800 p-3" onClick={()=>onFallback(error)}>Continue with Three.js</button><button onClick={onExit}>Back to dashboard</button></div>}
  </div>;
};
