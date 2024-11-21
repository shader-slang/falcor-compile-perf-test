/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "EmissiveLightSamplerType.slangh"
__exported import Rendering.Lights.EmissiveLightSamplerInterface;
__exported import Utils.Sampling.SampleGeneratorInterface;

/** The host sets the _EMISSIVE_LIGHT_SAMPLER_TYPE define to select sampler.

    This code typedefs the chosen type to the type 'EmissiveLightSampler'.
    All emissive samplers adhere to the same interface, but note that the
    size of the 'EmissiveLightSampler' type may vary depending on the type.
*/

// If _EMISSIVE_LIGHT_SAMPLER_TYPE is not defined, define it to the null sampler.
// This case happens if the user imports EmissiveLightSampler.slang but doesn't use it.
// This is a workaround to avoid compiler warnings since Slang warns on undefined
// preprocessor defines even if they are inside #ifdef .. #endif blocks.
#ifndef _EMISSIVE_LIGHT_SAMPLER_TYPE
#define _EMISSIVE_LIGHT_SAMPLER_TYPE EMISSIVE_LIGHT_SAMPLER_NULL
#endif

#if _EMISSIVE_LIGHT_SAMPLER_TYPE == EMISSIVE_LIGHT_SAMPLER_UNIFORM
    import Rendering.Lights.EmissiveUniformSampler;
    typedef EmissiveUniformSampler EmissiveLightSampler;

#elif _EMISSIVE_LIGHT_SAMPLER_TYPE == EMISSIVE_LIGHT_SAMPLER_LIGHT_BVH
    import Rendering.Lights.LightBVHSampler;
    typedef LightBVHSampler EmissiveLightSampler;

#elif _EMISSIVE_LIGHT_SAMPLER_TYPE == EMISSIVE_LIGHT_SAMPLER_POWER
    import Rendering.Lights.EmissivePowerSampler;
    typedef EmissivePowerSampler EmissiveLightSampler;

#elif _EMISSIVE_LIGHT_SAMPLER_TYPE == EMISSIVE_LIGHT_SAMPLER_NULL

    /** Null sampler.
        This is provided so that implementations always have a valid sampler struct.
    */
    struct NullEmissiveSampler : IEmissiveLightSampler
    {
        bool sampleLight<S : ISampleGenerator>(const float3 posW, const float3 normalW, const bool upperHemisphere, inout S sg, out TriangleLightSample ls)
        {
            ls = {};

            return false;
        }

        float evalTriangleSelectionPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const uint triangleIndex)
        {
            return 0.f;
        }

        float evalPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const TriangleLightHit hit)
        {
            return 0.f;
        }

        int _dummy; // Dummy variable to make sure Slang doesn't remove the struct.
    };
    typedef NullEmissiveSampler EmissiveLightSampler;

#else
    // Compile-time error if _EMISSIVE_LIGHT_SAMPLER_TYPE is an invalid type.
    #error _EMISSIVE_LIGHT_SAMPLER_TYPE is not set to a supported type. See EmissiveLightSamplerType.slangh.

#endif
