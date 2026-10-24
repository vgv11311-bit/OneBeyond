using System;
using System.Collections;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using GLTFast;
using GLTFast.Logging;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class OneBeyondViewer : MonoBehaviour
{
    [Serializable] public class SceneRequest {
        public string url, session;
        public float width, length, height, fy, photoAspect;
        public float[] position, forward, up;
        public Obstacle[] obstacles;
        public Fixture[] fixtures;
        public float exposure = 1, ambient = .5f;
    }
    [Serializable] public class Obstacle { public float[] center, size; public float yaw; }
    [Serializable] public class Fixture { public float[] position, color; public float intensity; public bool ceiling; }
    [Serializable] class ViewerEvent {
        public string type, session, message, quality;
        public float fps, frameMs;
    }
    [Serializable] class Motion { public float x, z, y; }
    [DllImport("__Internal")] static extern void OneBeyondEvent(string message);
    Camera view;
    GltfImport importer;
    CancellationTokenSource cancellation;
    GameObject room, lights;
    SceneRequest scene;
    string session = "";
    Vector3 startPosition;
    Quaternion startRotation;
    float yaw, pitch;
    int generation;
    bool loaded, focused, automatic = true;
    int tier = 1, stableWindows;
    float frameSeconds, sampleSeconds;
    int frames;
    Motion motion = new Motion();
    Cubemap reflection;
    int screenWidth, screenHeight;

    void Start() {
        view = Camera.main;
        Application.targetFrameRate = 60;
        #if UNITY_WEBGL && !UNITY_EDITOR
        WebGLInput.captureAllKeyboardInput=false;
        #endif
        reflection=new Cubemap(16,TextureFormat.RGBA32,true);
        var pixels=new Color[16*16];
        for(int face=0;face<6;face++) {
            for(int i=0;i<pixels.Length;i++) pixels[i]=Color.white*(face == 2 ? .65f : face == 3 ? .22f : .42f);
            reflection.SetPixels(pixels,(CubemapFace)face);
        }
        reflection.Apply(true,false);
        RenderSettings.defaultReflectionMode=DefaultReflectionMode.Custom;
        RenderSettings.customReflectionTexture=reflection;
        Emit("ready");
    }
    void Emit(string type, string message = "") {
        var value = JsonUtility.ToJson(new ViewerEvent { type=type,session=session,message=message,
            quality=tier==2?"high":tier==1?"balanced":"compatibility",fps=frames/Mathf.Max(.001f,frameSeconds),
            frameMs=1000f*frameSeconds/Mathf.Max(1,frames) });
        #if UNITY_WEBGL && !UNITY_EDITOR
        OneBeyondEvent(value);
        #else
        Debug.Log(value);
        #endif
    }
    public async void LoadScene(string json) {
        var current = ++generation;
        cancellation?.Cancel(); cancellation?.Dispose();
        cancellation = new CancellationTokenSource();
        loaded = false; scene = JsonUtility.FromJson<SceneRequest>(json); session=scene.session;
        if (room) Destroy(room);
        importer?.Dispose(); importer=null;
        if (lights) Destroy(lights);
        var next = new GltfImport(logger:new ConsoleLogger());
        try {
            if (!Uri.TryCreate(scene.url, UriKind.Absolute, out var uri) || (uri.Scheme!="http" && uri.Scheme!="https"))
                throw new Exception("Only HTTP(S) room URLs are supported.");
            // Download once so unsupported optional emission metadata can be
            // preserved without a second network request for the same model.
            var token = cancellation.Token;
            using var download = UnityWebRequest.Get(uri);
            download.timeout = 90;
            var operation = download.SendWebRequest();
            while (!operation.isDone) { token.ThrowIfCancellationRequested(); await Task.Yield(); }
            token.ThrowIfCancellationRequested();
            if (download.result != UnityWebRequest.Result.Success) throw new Exception("The reconstructed GLB download failed.");
            var bytes = download.downloadHandler.data;
            var emission = GlbEmission.Read(bytes);
            var success = await next.LoadGltfBinary(bytes, uri, cancellationToken:token);
            if (current != generation) { next.Dispose(); return; }
            if (!success) throw new Exception("The reconstructed GLB could not be loaded.");
            var emissionCount = GlbEmission.Apply(next, emission);
            if (emissionCount > 0) Emit("materials", "Preserved emissive strength for " + emissionCount + " material(s).");
            var candidate = new GameObject("ReconstructedRoom");
            if (!await next.InstantiateMainSceneAsync(candidate.transform)) {Destroy(candidate);throw new Exception("GLB scene could not be instantiated.");}
            if (current != generation) {Destroy(candidate);next.Dispose();return;}
            room=candidate;
            importer=next;
            // glTFast mirrors X to convert glTF's right-handed frame to Unity.
            // Camera, obstacles and fixtures arrive in that same converted frame.
            foreach (var c in room.GetComponentsInChildren<Camera>()) c.enabled=false;
            foreach (var l in room.GetComponentsInChildren<Light>()) l.enabled=false;
            foreach (var r in room.GetComponentsInChildren<Renderer>()) {
                r.shadowCastingMode = ShadowCastingMode.On; r.receiveShadows=true;
            }
            startPosition=Vector(scene.position);
            startRotation=Quaternion.LookRotation(Vector(scene.forward),Vector(scene.up));
            ResetCamera(""); ConfigureLighting("cinematic"); ResizeLens();
            loaded=true; frameSeconds=0; frames=0; sampleSeconds=0; stableWindows=0;
            Emit("loaded");
        } catch (Exception e) {
            next.Dispose();
            if (current==generation) Emit("error", e.Message);
        }
    }
    static Vector3 Vector(float[] p) => p!=null && p.Length==3 ? new Vector3(p[0],p[1],p[2]) : Vector3.zero;
    public void ResetCamera(string _) {
        view.transform.SetPositionAndRotation(startPosition,startRotation);
        var e=view.transform.eulerAngles; yaw=e.y; pitch=e.x>180?e.x-360:e.x;
    }
    public void SetQuality(string quality) {
        automatic=quality=="auto";
        // Never infer GPU power from core count alone. Begin conservatively,
        // then use sustained frame-time windows with hysteresis.
        tier=quality=="high"?2:quality=="compatibility"?0:1;
        ApplyQuality(); stableWindows=0; frameSeconds=0; frames=0; sampleSeconds=0;
    }
    void ApplyQuality() {
        var pipeline=GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
        if (pipeline) {
            pipeline.renderScale=tier == 0 ? .7f : tier == 1 ? .9f : 1f;
            pipeline.msaaSampleCount=tier==2?4:2;
            pipeline.shadowDistance=tier==0?0:12;
        }
        QualitySettings.globalTextureMipmapLimit=tier==0?1:0;
        Emit("quality");
    }
    public void SetLighting(string mode) { if (scene!=null) ConfigureLighting(mode); }
    void ConfigureLighting(string mode) {
        if (lights) Destroy(lights);
        lights=new GameObject("EvidenceLighting");
        RenderSettings.ambientMode=AmbientMode.Trilight;
        float strength=mode == "studio" ? .8f : mode == "daylight" ? .65f : Mathf.Clamp(scene.ambient,.25f,.7f);
        RenderSettings.ambientSkyColor=Color.white*strength;
        RenderSettings.ambientEquatorColor=Color.white*strength*.8f;
        RenderSettings.ambientGroundColor=Color.white*strength*.5f;
        // Runtime-loaded rooms have no baked light probe. Explicitly provide
        // neutral diffuse irradiance; changing ambient colors alone can leave
        // the URP shader's spherical-harmonic probe empty in a Web player.
        var probe = new SphericalHarmonicsL2();
        probe.AddAmbientLight(Color.white * strength * .7f);
        RenderSettings.ambientProbe = probe;
        RenderSettings.reflectionIntensity=.35f;
        var key=new GameObject("NeutralFill").AddComponent<Light>();
        key.transform.SetParent(lights.transform); key.type=LightType.Directional;
        key.transform.rotation=Quaternion.Euler(45,-30,0); key.color=Color.white;
        key.intensity=mode == "studio" ? .65f : .4f; key.shadows=LightShadows.Soft;
        key.shadowBias=.05f; key.shadowNormalBias=.25f;
        if (scene.fixtures!=null) foreach (var fixture in scene.fixtures) {
            var light=new GameObject("PhotoFixture").AddComponent<Light>();
            light.transform.SetParent(lights.transform);light.transform.position=Vector(fixture.position);
            var color=Vector(fixture.color); light.color=new Color(color.x,color.y,color.z);
            light.type=fixture.ceiling?LightType.Spot:LightType.Point;
            light.transform.rotation=Quaternion.Euler(90,0,0);light.spotAngle=110;
            light.intensity=Mathf.Clamp(fixture.intensity,0,8); light.range=5; light.shadows=LightShadows.None;
        }
        var volume=FindFirstObjectByType<Volume>();
        if (volume && volume.profile.TryGet<ColorAdjustments>(out var adjust))
            adjust.postExposure.Override(Mathf.Log(Mathf.Clamp(scene.exposure,.5f,2),2));
    }
    public void SetFocused(string value) { focused=value=="true"; if(!focused) motion=new Motion(); }
    public void SetMotion(string value) { motion=JsonUtility.FromJson<Motion>(value); }
    public void Look(string value) {
        if(!loaded || !focused) return;
        var delta=JsonUtility.FromJson<Motion>(value);
        yaw+=delta.x*.126f;pitch=Mathf.Clamp(pitch+delta.y*.126f,-85,85);
        view.transform.rotation=Quaternion.Euler(pitch,yaw,0);
    }
    public void Capture(string _) { StartCoroutine(Screenshot()); }
    IEnumerator Screenshot() {
        yield return new WaitForEndOfFrame();
        var image=ScreenCapture.CaptureScreenshotAsTexture();
        Emit("screenshot",Convert.ToBase64String(image.EncodeToPNG())); Destroy(image);
    }
    void ResizeLens() {
        screenWidth=Screen.width;screenHeight=Screen.height;
        float aspect=(float)screenWidth/Mathf.Max(1,screenHeight);
        view.fieldOfView=scene!=null && scene.fy>0 ? 2*Mathf.Atan(.5f/scene.fy*Mathf.Max(1,scene.photoAspect/aspect))*Mathf.Rad2Deg : 68;
    }
    void Update() {
        if (!loaded) return;
        if (screenWidth!=Screen.width || screenHeight!=Screen.height) ResizeLens();
        if (focused) {
            float x=(Input.GetKey(KeyCode.D)?1:0)-(Input.GetKey(KeyCode.A)?1:0)+motion.x;
            float z=(Input.GetKey(KeyCode.W)?1:0)-(Input.GetKey(KeyCode.S)?1:0)+motion.z;
            float y=(Input.GetKey(KeyCode.Space)?1:0)-(Input.GetKey(KeyCode.C)?1:0)+motion.y;
            var forward=Vector3.ProjectOnPlane(view.transform.forward,Vector3.up).normalized;
            var right=Vector3.ProjectOnPlane(view.transform.right,Vector3.up).normalized;
            var delta=(forward*z+right*x+Vector3.up*y);
            if (delta.magnitude>1) delta.Normalize();
            Move(delta*(Input.GetKey(KeyCode.LeftShift)?5:2.5f)*Mathf.Min(Time.unscaledDeltaTime,.05f));
        }
        if (!Application.isFocused) return;
        frames++;frameSeconds+=Time.unscaledDeltaTime;sampleSeconds+=Time.unscaledDeltaTime;
        if (sampleSeconds>=5) {
            Emit("metrics");
            float fps=frames/Mathf.Max(.001f,frameSeconds);
            if(automatic) {
                if(fps<27 && tier>0) {tier--;stableWindows=0;ApplyQuality();}
                else if(fps>55 && tier<2 && ++stableWindows>=3) {tier++;stableWindows=0;ApplyQuality();}
                else if(fps<=55) stableWindows=0;
            }
            frames=0;frameSeconds=0;sampleSeconds=0;
        }
    }
    bool Blocked(Vector3 p,Obstacle o) {
        var c=Vector(o.center);var size=Vector(o.size);
        if(p.y-.65f>=c.y+size.z/2 || p.y+.12f<=c.y-size.z/2) return false;
        float angle=o.yaw*Mathf.Deg2Rad,dx=p.x-c.x,dz=p.z-c.z;
        return Mathf.Abs(Mathf.Cos(angle)*dx-Mathf.Sin(angle)*dz)<size.x/2+.18f && Mathf.Abs(Mathf.Sin(angle)*dx+Mathf.Cos(angle)*dz)<size.y/2+.18f;
    }
    void Move(Vector3 delta) {
        var p=view.transform.position;
        int steps=Mathf.Max(1,Mathf.CeilToInt(delta.magnitude/.06f));
        for(int step=0;step<steps;step++) foreach(int axis in new[]{0,2,1}) {
            var next=p;next[axis]+=delta[axis]/steps;
            next.x=Mathf.Clamp(next.x,-scene.width/2+.22f,scene.width/2-.22f);
            next.z=Mathf.Clamp(next.z,-scene.length+.32f,.02f);next.y=Mathf.Clamp(next.y,.4f,scene.height-.2f);
            bool blocked=false;
            if(scene.obstacles!=null) foreach(var o in scene.obstacles) if(!Blocked(p,o)&&Blocked(next,o)) {blocked=true;break;}
            if(!blocked) p=next;
        }
        view.transform.position=p;
    }
    void OnApplicationFocus(bool value) { if(!value) {focused=false;motion=new Motion();} }
    void OnDestroy() {generation++;cancellation?.Cancel();cancellation?.Dispose();importer?.Dispose();if(reflection)Destroy(reflection);}
}
