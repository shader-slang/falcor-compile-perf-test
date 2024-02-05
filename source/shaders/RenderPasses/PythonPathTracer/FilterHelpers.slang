/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.TexLODHelpers;
import Utils.Math.MathHelpers;
import Scene.Scene;

cbuffer FilterCB
{
    float gSpatialPerturbationScale;
    float gScreenSpacePixelSpreadAngle;
    bool gPrefilterIsotropic;
};

void computeFilterEllipse(VertexData v, StaticVertexData vertices[3], float3 rayDir, float hitT, out float2 ellipseX, out float2 ellipseY)
{
    float rayConeWidth = hitT * gScreenSpacePixelSpreadAngle;
    float2 txcoords[3] = { vertices[0].texCrd, vertices[1].texCrd, vertices[2].texCrd };
    float3 positions[3] = { vertices[0].position, vertices[1].position, vertices[2].position };

    computeAnisotropicEllipseAxes(v.posW, v.normalW, rayDir, rayConeWidth, positions, txcoords, v.texC, ellipseX, ellipseY);

    if (gPrefilterIsotropic)
    {
        // Convert ellipse to equal area circle
        float radius = sqrt(length(ellipseX) * length(ellipseY));
        // Use maximum of either filter footprint
        // float radius = max(length(ellipseX), length(ellipseY));
        ellipseX = normalize(ellipseX) * radius;
        ellipseY = normalize(ellipseY) * radius;
    }
}

float2 sampleFilterFootprint(float2 ellipseX, float2 ellipseY, float2 center, inout SampleGenerator sg)
{
    float2 xi = sampleNext2D(sg);
    float r = sqrt(-2.0f * log(xi.x)) * 0.65f;
    float phi = xi.y * 2.0f * M_PI;
    float2 xy = float2(cos(phi) * r, sin(phi) * r) * gSpatialPerturbationScale * 0.5f;
    // float2 xy = sample_disk(sampleNext2D(sg))*gSpatialPerturbationScale*0.5f;

    return center + ellipseX * xy.x + ellipseY * xy.y;
}

void sampleFilterFootprint(float2 ellipseX, float2 ellipseY, inout VertexData v, inout SampleGenerator sg)
{
    v.texC = sampleFilterFootprint(ellipseX, ellipseY, v.texC, sg);
}
