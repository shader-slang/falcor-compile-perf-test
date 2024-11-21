/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Sampling.SampleGeneratorInterface;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.EmissiveLightSamplerInterface;

struct EmissivePower
{
    float           invWeightsSum;
    Buffer<uint2>   triangleAliasTable;
};

/** Emissive light sampler that samples proportionally to emissive power.

    The sampler implements the IEmissiveLightSampler interface (see
    EmissiveLightSamplerInterface.slang for usage information).

    The struct wraps a LightCollection that stores the pre-processed lights.
    The program should instantiate the struct below. See EmissiveLightSampler.slang.
*/
struct EmissivePowerSampler : IEmissiveLightSampler
{
    EmissivePower    _emissivePower;

    /** Draw a single light sample.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Light sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleLight<S : ISampleGenerator>(const float3 posW, const float3 normalW, const bool upperHemisphere, inout S sg, out TriangleLightSample ls)
    {
        ls = {};

        if (gScene.lightCollection.isEmpty()) return false;

        // Randomly pick a triangle out of the global list with uniform probability.
        float uLight = sampleNext1D(sg);
        uint triangleCount = gScene.lightCollection.triangleCount;
        // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
        uint triangleIndex = min((uint)(uLight * triangleCount), triangleCount - 1);

        uint2 packed = _emissivePower.triangleAliasTable[triangleIndex];
        float threshold = f16tof32(packed.x >> 16u);
        uint  selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
        uint  selectBelow = packed.y & 0xFFFFFFu;

        // Test the threshold in the current table entry; pick one of the two options
        triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

        float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * _emissivePower.invWeightsSum;

        // Sample the triangle uniformly.
        float2 u = sampleNext2D(sg);

        if (!sampleTriangle(posW, triangleIndex, u, ls)) return false;

        // The final probability density is the product of the sampling probabilities.
        ls.pdf *= triangleSelectionPdf;
        return true;
    }

    /** Evaluate the PDF associated with selecting an emissive triangle.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in] triangleIndex Index of selected triangle.
        \return Probability of drawing the input triangle proportionally to its emissive power
    */
    float evalTriangleSelectionPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const uint triangleIndex)
    {
        if (triangleIndex == LightCollection::kInvalidIndex) return 0.f;

        return gScene.lightCollection.fluxData[triangleIndex].flux * _emissivePower.invWeightsSum;
    }

    /** Evaluate the PDF at a shading point given a hit point on an emissive triangle.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] hit Triangle hit data.
        \return Probability density with respect to solid angle at the shading point.
    */
    float evalPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const TriangleLightHit hit)
    {
        if (hit.triangleIndex == LightCollection::kInvalidIndex) return 0.f;

        float triangleSelectionPdf = evalTriangleSelectionPdf(posW, normalW, upperHemisphere, hit.triangleIndex);

        // Compute triangle sampling probability with respect to solid angle from the shading point.
        float trianglePdf = evalTrianglePdf(posW, hit);

        // The final probability density is the product of the sampling probabilities.
        return triangleSelectionPdf * trianglePdf;
    }
};
