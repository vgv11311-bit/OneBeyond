using System;
using System.Text;
using GLTFast;
using UnityEngine;

// glTFast 6.20 imports emissiveFactor but not KHR_materials_emissive_strength.
// Read only that extension from the same downloaded GLB; never infer brightness
// from object names or reference-photo-specific material settings.
public static class GlbEmission
{
    [Serializable] public class Root { public Entry[] materials; }
    [Serializable] public class Entry { public Extensions extensions; }
    [Serializable] public class Extensions { public Strength KHR_materials_emissive_strength; }
    [Serializable] public class Strength { public float emissiveStrength = 1; }

    public static Root Read(byte[] bytes)
    {
        if (bytes == null || bytes.Length < 20 || BitConverter.ToUInt32(bytes, 0) != 0x46546c67 ||
            BitConverter.ToUInt32(bytes, 4) != 2 || BitConverter.ToUInt32(bytes, 8) != bytes.Length ||
            BitConverter.ToUInt32(bytes, 16) != 0x4e4f534a) return null;
        uint length = BitConverter.ToUInt32(bytes, 12);
        if (length > bytes.Length - 20) return null;
        return JsonUtility.FromJson<Root>(Encoding.UTF8.GetString(bytes, 20, (int)length));
    }

    public static int Apply(GltfImport importer, Root root)
    {
        int count = 0;
        if (root?.materials == null) return count;
        for (int i = 0; i < root.materials.Length; i++) {
            var extension = root.materials[i]?.extensions?.KHR_materials_emissive_strength;
            // JsonUtility constructs absent nested objects with their default
            // value (1). A missing extension must therefore be a no-op.
            if (extension == null || extension.emissiveStrength == 1 || float.IsNaN(extension.emissiveStrength) ||
                float.IsInfinity(extension.emissiveStrength) || extension.emissiveStrength < 0) continue;
            var material = importer.GetMaterial(i);
            // glTFast URP Shader Graph's linear-space emission input.
            if (material && material.HasProperty("emissiveFactor")) {
                material.SetColor("emissiveFactor", material.GetColor("emissiveFactor") * extension.emissiveStrength);
                count++;
            }
        }
        return count;
    }
}
