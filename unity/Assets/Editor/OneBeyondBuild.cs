using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public static class OneBeyondBuild
{
    // Rebuild the player only when viewer code changes, never for a room upload.
    public static void Web()
    {
        GlbEmissionChecks.Run();
        Directory.CreateDirectory("Assets/Generated");
        var renderer = AssetDatabase.LoadAssetAtPath<UniversalRendererData>("Assets/Generated/Renderer.asset");
        if (!renderer) {
            renderer = ScriptableObject.CreateInstance<UniversalRendererData>();
            AssetDatabase.CreateAsset(renderer, "Assets/Generated/Renderer.asset");
        }
        var pipeline = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>("Assets/Generated/Pipeline.asset");
        if (!pipeline) {
            pipeline = UniversalRenderPipelineAsset.Create(renderer);
            AssetDatabase.CreateAsset(pipeline, "Assets/Generated/Pipeline.asset");
        }
        pipeline.name = "OneBeyondURP";
        pipeline.renderScale = 1f;
        pipeline.msaaSampleCount = 4;
        pipeline.shadowDistance = 12f;
        pipeline.supportsHDR = true;
        var settings = new SerializedObject(pipeline);
        settings.FindProperty("m_MainLightShadowsSupported").boolValue = true;
        settings.FindProperty("m_AdditionalLightsRenderingMode").intValue = 1;
        settings.FindProperty("m_AdditionalLightShadowsSupported").boolValue = false;
        settings.FindProperty("m_SoftShadowsSupported").boolValue = true;
        settings.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(pipeline);
        GraphicsSettings.defaultRenderPipeline = pipeline;
        QualitySettings.renderPipeline = pipeline;
        IncludeRuntimeShaders();
        QualitySettings.vSyncCount = 0;
        PlayerSettings.companyName = "One Beyond";
        PlayerSettings.productName = "One Beyond Spatial Viewer";
        PlayerSettings.colorSpace = ColorSpace.Linear;
        PlayerSettings.runInBackground = false;
        PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Gzip;
        // Unity's loader decompresses .unityweb on static hosts without special headers.
        PlayerSettings.WebGL.decompressionFallback = true;
        PlayerSettings.WebGL.dataCaching = true;
        PlayerSettings.WebGL.initialMemorySize = 128;
        PlayerSettings.WebGL.maximumMemorySize = 1024;
        PlayerSettings.stripEngineCode = true;
        PlayerSettings.SetManagedStrippingLevel(UnityEditor.Build.NamedBuildTarget.WebGL, ManagedStrippingLevel.Low);
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        new GameObject("OneBeyondViewer").AddComponent<OneBeyondViewer>();
        var camera = new GameObject("Camera").AddComponent<Camera>();
        camera.tag = "MainCamera";
        camera.nearClipPlane = .05f;
        camera.farClipPlane = 100f;
        camera.backgroundColor = new Color(.06f,.06f,.06f);
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.gameObject.AddComponent<UniversalAdditionalCameraData>().renderPostProcessing = true;
        var volume = new GameObject("ToneMapping").AddComponent<Volume>();
        volume.isGlobal = true;
        var profile = AssetDatabase.LoadAssetAtPath<VolumeProfile>("Assets/Generated/Volume.asset");
        if (!profile) {
            profile = ScriptableObject.CreateInstance<VolumeProfile>();
            AssetDatabase.CreateAsset(profile, "Assets/Generated/Volume.asset");
            var tone=profile.Add<Tonemapping>(true);tone.mode.Override(TonemappingMode.ACES);
            var adjust=profile.Add<ColorAdjustments>(true);adjust.postExposure.Override(0);
            AssetDatabase.AddObjectToAsset(tone,profile);AssetDatabase.AddObjectToAsset(adjust,profile);
            EditorUtility.SetDirty(profile);
        }
        volume.sharedProfile = profile;
        EditorSceneManager.SaveScene(scene, "Assets/Generated/Viewer.unity");
        AssetDatabase.SaveAssets();
        var output = Path.GetFullPath(Path.Combine(Application.dataPath, "../../frontend/public/unity"));
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = new[]{"Assets/Generated/Viewer.unity"}, locationPathName = output,
            target = BuildTarget.WebGL, options = BuildOptions.None
        });
        if (report.summary.result != BuildResult.Succeeded) throw new Exception("Unity web build failed: " + report.summary.result);
        string FileUrl(string pattern) => "Build/" + Path.GetFileName(Directory.GetFiles(Path.Combine(output,"Build"),pattern).Single());
        var config = new BuildManifest { loaderUrl = FileUrl("*.loader.js"), dataUrl = FileUrl("*.data*"),
            frameworkUrl = FileUrl("*.framework.js*"), codeUrl = FileUrl("*.wasm*"), buildId = DateTime.UtcNow.ToString("yyyyMMddHHmmss") };
        File.WriteAllText(Path.Combine(output, "viewer.json"), JsonUtility.ToJson(config));
        Debug.Log("ONE_BEYOND_WEB_BUILD_OK " + output);
    }
    [Serializable] class BuildManifest { public int version = 1; public string loaderUrl, dataUrl, frameworkUrl, codeUrl, buildId; }

    static void IncludeRuntimeShaders()
    {
        // Runtime-imported GLBs have no scene material references at build time.
        // Preserve glTFast's shaders instead of silently shipping pink/missing
        // or feature-stripped materials. Narrow variants only after profiling.
        var shaders=AssetDatabase.GetAllAssetPaths()
            .Where(p=>p.StartsWith("Packages/com.unity.cloud.gltfast/Runtime/Shader/",StringComparison.Ordinal)
                && !p.Contains("/HDRP/") && !p.Contains("/Built-In/")
                && (p.EndsWith(".shadergraph") || p.EndsWith(".shader")))
            .Select(AssetDatabase.LoadAssetAtPath<Shader>).Where(s=>s!=null).Distinct().ToArray();
        if(shaders.Length==0) throw new Exception("No glTFast runtime shaders resolved; refusing to build an unusable viewer.");
        var graphics=new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset")[0]);
        var included=graphics.FindProperty("m_AlwaysIncludedShaders");
        foreach(var shader in shaders) {
            bool exists=false;
            for(int i=0;i<included.arraySize;i++) if(included.GetArrayElementAtIndex(i).objectReferenceValue==shader) {exists=true;break;}
            if(!exists) {int i=included.arraySize;included.InsertArrayElementAtIndex(i);included.GetArrayElementAtIndex(i).objectReferenceValue=shader;}
        }
        graphics.ApplyModifiedPropertiesWithoutUndo();
    }
}
