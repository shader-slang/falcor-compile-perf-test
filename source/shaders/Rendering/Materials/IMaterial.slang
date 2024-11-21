/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterialInstance;
__exported import Scene.Material.MaterialSystem;
__exported import Scene.Material.TextureSampler;
__exported import Scene.Material.VolumeProperties;
import Rendering.Volumes.PhaseFunction;
__exported import Scene.ShadingData;

/** Interface for materials.

    Implementations of this interface hold all data describing the material properties.
    Material instances are created and initialized with data by `MaterialSystem`.

    When querying a material at a given shading point, a material instance implementing
    the IMaterialInstance interface is returned. This has all material properties evaluated at the shading point.
    The process of creating this instance is referred to as "pattern generation". The returned
    instance can be used for evaluation and sampling of the BSDF, and associated functionality.

    Note:
     - The [anyValueSize(n)] attribute specifies the maximum size in bytes an implementation type
       to `IMaterial` can be. Slang compiler will check the implementation types and emits an error
       if an implementation exceeds this size.
     - The value set here should correspond to the size of `MaterialDataBlob` (128B).
*/
[anyValueSize(128)]
interface IMaterial
{
    // Requires an implementation to define a type named `MaterialInstance` that
    // conforms to the `IMaterialInstance` interface.
    associatedtype MaterialInstance : IMaterialInstance;

    /** Return a MaterialInstance for the given shading point.
        This operation performs pattern generation and may sample textures etc.
        \param[in] ms Material system. This provides the resources for pattern generation.
        \param[in] sd Shading data.
        \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
        \param[in] hints Optional hints to control the behavior of the material instance. Combination of `MaterialInstanceHints` flags.
        \return MaterialInstance implementing the `IMaterialInstance` interface.
    */
    MaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints = (uint)MaterialInstanceHints::None);

    /** A differentiable version of `setupMaterialInstance`.
        This operation returns an additional `DiffMaterialData` structure that contains differentiable parameters.
    */
    [Differentiable]
    MaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints = (uint)MaterialInstanceHints::None);

    /** Returns volume properties that are homogeneous for the whole volume (do not depend on the hitpoint).
     */
    VolumeProperties getHomogeneousVolumeProperties(const MaterialSystem ms, const uint materialID);
}

[open]
struct MaterialBase
{
    MaterialHeader header;

    VolumeProperties getHomogeneousVolumeProperties(const MaterialSystem ms, const uint materialID)
    {
        if (header.isBasicMaterial())
        {
            const BasicMaterialData md = ms.getBasicMaterialData(materialID);
            return { md.volumeAbsorption, md.volumeScattering, HenyeyGreensteinPhaseFunction(md.volumeAnisotropy) };
        }
        return { float3(0.f), float3(0.f), NullPhaseFunction() };
    }
}
