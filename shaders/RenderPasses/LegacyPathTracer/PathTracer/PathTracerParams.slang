/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"
#include "Rendering/Lights/EmissiveLightSamplerType.slangh"

#ifdef HOST_CODE
#include "Utils/Properties.h"
#endif

BEGIN_NAMESPACE_FALCOR

// Define max supported path length and number of shadow rays per vertex.
// These limits depend on the bit layout for the packed path flags.
static const uint kMaxPathFlagsBits = 16;
static const uint kMaxPathLengthBits = 8;
static const uint kMaxPathLength = (1 << kMaxPathLengthBits) - 1;
static const uint kMaxLightSamplesPerVertex = 8;
static const uint kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to
                                         // avoid getting stuck in pathological cases.

// Define ray indices.
static const uint32_t kRayTypeScatter = 0;
static const uint32_t kRayTypeShadow = 1;

// TODO: Remove explicitly assigned values when we can use the enums as default initializer below
enum class MISHeuristic
{
    BalanceHeuristic = 0,  ///< Balance heuristic.
    PowerTwoHeuristic = 1, ///< Power heuristic (exponent = 2.0).
    PowerExpHeuristic = 2, ///< Power heuristic (variable exponent).
};

/** Path tracer parameters. Shared between host and device.

    Note that if you add configuration parameters, do not forget to register
    them with the scripting system in FALCOR_SCRIPT_BINDING() in PathTracer.cpp.
*/
struct PathTracerParams
{
    // Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
    // Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
    // https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
    // Note that the default initializers are ignored by Slang but used on the host.

    // clang-format off
    // General
    uint    samplesPerPixel = 1;            ///< Number of samples (paths) per pixel. Use compile-time constant kSamplesPerPixel in shader.
    uint    lightSamplesPerVertex = 1;      ///< Number of light samples per path vertex. Use compile-time constant kLightSamplesPerVertex in shader.
    uint    maxBounces = 3;                 ///< Max number of indirect bounces (0 = none), up to kMaxPathLength. Use compile-time constant kMaxBounces in shader.
    uint    maxNonSpecularBounces = 3;      ///< Max number of non-specular indirect bounces (0 = none), up to kMaxPathLength. Use compile-time constant kMaxNonSpecularBounces in shader.

    int     useVBuffer = 1;                 ///< Use a V-buffer as input. Use compile-time constant kUseVBuffer (or preprocessor define USE_VBUFFER) in shader.
    int     useAlphaTest = 1;               ///< Use alpha testing on non-opaque triangles. Use compile-time constant kUseAlphaTest (or preprocessor define USE_ALPHA_TEST) in shader.
    int     adjustShadingNormals = false;   ///< Adjust shading normals on secondary hits. Use compile-time constant kAdjustShadingNormals in shader.
    int     forceAlphaOne = true;           ///< Force the alpha channel to 1.0. Otherwise background will have alpha 0.0 and covered samples 1.0 to allow compositing. Use compile-time constant kForceAlphaOne in shader.

    int     clampSamples = false;           ///< Clamp the per-path contribution to 'clampThreshold' to reduce fireflies.
    float   clampThreshold = 10.f;
    float   specularRoughnessThreshold = 0.25f; ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold.
    float   _pad0;

    // Sampling
    int     useBRDFSampling = true;         ///< Use BRDF importance sampling (otherwise cosine-weighted hemisphere sampling). Use compile-time constant kUseBRDFSampling in shader.
    int     useNEE = true;                  ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
    int     useMIS = true;                  ///< Use multiple importance sampling (MIS) when NEE is enabled. This enables a scatter ray from the last path vertex. Use compile-time constant kUseMIS in shader.
    uint    misHeuristic = 1; /* (uint)MISHeuristic::PowerTwoHeuristic */   ///< MIS heuristic. Use compile-time constant kMISHeuristic in shader. TODO: Replace initializer value by enum when Slang supports it.

    float   misPowerExponent = 2.f;         ///< MIS exponent for the power heuristic. This is only used when 'PowerExpHeuristic' is chosen.
    int     useRussianRoulette = false;     ///< Use Russian roulette. Use compile-time constant kUseRussianRoulette in shader.
    float   probabilityAbsorption = 0.2f;   ///< Probability of absorption for Russian roulette.
    int     useFixedSeed = false;           ///< Use fixed random seed for the sample generator. This is useful for print() debugging.

    int     useNestedDielectrics = true;    ///< Use algorithm to handle nested dielectric materials. Use compile-time constant kUseNestedDielectrics in shader.
    int     useLightsInDielectricVolumes = false; ///< Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface. Use compile-time constant kUseLightsInDielectricVolumes in shader.
    int     disableCaustics = false;        ///< Disable sampling of caustics. Use compile-time constant kDisableCaustics in shader.
    int     _pad1;

    // Ray footprint
    float   screenSpacePixelSpreadAngle = 0.0; ///< The angle an "average" pixel spans from camera (Used by ray footprint).
    uint    rayFootprintMode = 0;           ///< Ray footprint tracking mode used for Texture LOD. See RayFootprintModes.slangh.
    uint    rayConeMode = 0;                ///< Sub-mode for the Ray Cone mode of ray footprint. See TexLODTypes.slang.
    int     rayFootprintUseRoughness = 0;   ///< Integrates material roughness into the footprint calculation (only used by Ray Cone right now).

    // Runtime data
    uint2   frameDim = uint2(0, 0);         ///< Current frame dimensions in pixels.
    uint    frameCount = 0;                 ///< Frame count since scene was loaded.
    uint    prngDimension = 0;              ///< First available PRNG dimension.
    // clang-format on

#ifdef HOST_CODE
    template<typename Archive>
    void serialize(Archive& ar)
    {
        ar("samplesPerPixel", samplesPerPixel);
        ar("lightSamplesPerVertex", lightSamplesPerVertex);
        ar("maxBounces", maxBounces);
        ar("maxNonSpecularBounces", maxNonSpecularBounces);

        ar("useVBuffer", useVBuffer);
        ar("useAlphaTest", useAlphaTest);
        ar("adjustShadingNormals", adjustShadingNormals);
        ar("forceAlphaOne", forceAlphaOne);

        ar("clampSamples", clampSamples);
        ar("clampThreshold", clampThreshold);
        ar("specularRoughnessThreshold", specularRoughnessThreshold);

        ar("useBRDFSampling", useBRDFSampling);
        ar("useNEE", useNEE);
        ar("useMIS", useMIS);
        ar("misHeuristic", misHeuristic);

        ar("misPowerExponent", misPowerExponent);
        ar("useRussianRoulette", useRussianRoulette);
        ar("probabilityAbsorption", probabilityAbsorption);
        ar("useFixedSeed", useFixedSeed);

        ar("useNestedDielectrics", useNestedDielectrics);
        ar("useLightsInDielectricVolumes", useLightsInDielectricVolumes);
        ar("disableCaustics", disableCaustics);

        ar("rayFootprintMode", rayFootprintMode);
        ar("rayConeMode", rayConeMode);
        ar("rayFootprintUseRoughness", rayFootprintUseRoughness);
    }
#endif
};

END_NAMESPACE_FALCOR
