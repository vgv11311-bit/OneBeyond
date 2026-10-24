using System;
using System.Text;

public static class GlbEmissionChecks
{
    public static void Run()
    {
        var json = Encoding.UTF8.GetBytes("{\"materials\":[{}, {\"extensions\":{\"KHR_materials_emissive_strength\":{\"emissiveStrength\":4}}},{\"extensions\":{\"KHR_materials_emissive_strength\":{\"emissiveStrength\":0}}}]}");
        var bytes = new byte[20 + json.Length];
        BitConverter.GetBytes(0x46546c67u).CopyTo(bytes, 0);
        BitConverter.GetBytes(2u).CopyTo(bytes, 4);
        BitConverter.GetBytes((uint)bytes.Length).CopyTo(bytes, 8);
        BitConverter.GetBytes((uint)json.Length).CopyTo(bytes, 12);
        BitConverter.GetBytes(0x4e4f534au).CopyTo(bytes, 16);
        json.CopyTo(bytes, 20);
        var root = GlbEmission.Read(bytes);
        if (root?.materials?.Length != 3 || (root.materials[0]?.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? 1) != 1 ||
            root.materials[1].extensions.KHR_materials_emissive_strength.emissiveStrength != 4 ||
            root.materials[2].extensions.KHR_materials_emissive_strength.emissiveStrength != 0)
            throw new Exception("GLB emission metadata regression: " + UnityEngine.JsonUtility.ToJson(root));
        BitConverter.GetBytes(uint.MaxValue).CopyTo(bytes, 12);
        if (GlbEmission.Read(bytes) != null || GlbEmission.Read(new byte[0]) != null)
            throw new Exception("Truncated GLB validation regression.");
        UnityEngine.Debug.Log("ONE_BEYOND_EMISSION_CHECKS_OK");
    }
}
