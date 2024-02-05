/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;

struct BSDFScatterSample : IDifferentiable
{
    float3 wo;
    float pdf;
    float3 weight;
};

struct BSDFParameters : IDifferentiable
{
    float3 albedo;
    float roughness;
};


[Differentiable]
[PreferRecompute]
bool bsdfGGXSample(in ShadingData sd, in BSDFParameters params, inout SampleGenerator sg, out BSDFScatterSample result)
{
    result = {};

    float3 wiLocal = sd.frame.toLocal(sd.V);
    float2 u = no_diff sampleNext2D(sg);

    // Reject incoming directions that are too grazing.
    if (wiLocal.z < kMinCosTheta)
    {
        return false;
    }

    float pdf;
    float3 hLocal = no_diff sampleGGX_NDF(params.roughness, u, pdf);
    float wiDotH = detach(dot(wiLocal, hLocal));
    float3 woLocal = detach(2 * hLocal * wiDotH - wiLocal); // outgoing vector local space

    pdf = evalNdfGGX(params.roughness, hLocal.z) / (4.f * wiDotH);
    result.wo = sd.frame.fromLocal(woLocal); // wo to world.
    result.pdf = detach(pdf);
    result.weight = evalGGXDivByPDF(wiLocal, woLocal, hLocal, params) * pdf / detach(pdf);

    // Reject outgoing directions that are too grazing or outside the hemisphere.
    return woLocal.z > kMinCosTheta;
}

[Differentiable]
[PreferRecompute]
float3 evalGGXDivByPDF(in float3 wi, in float3 wo, in float3 h, in BSDFParameters params)
{
    float3 F = evalFresnelSchlick(params.albedo, float3(1.f), dot(wi, h));
    float G = evalMaskingSmithGGXCorrelated(params.roughness, wi.z, wo.z);
    return F * G * dot(wi, h) / max(1e-6f, wi.z * h.z);
}
