/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.MathHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

RWStructuredBuffer<float> result;
StructuredBuffer<float> input;

[numthreads(1024, 1, 1)]
void testSphericalCoordinates(uint3 threadId: SV_DispatchThreadID)
{
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(0), threadId.x);
    float3 dir = normalize(sampleNext3D(sg) * 2.f - 1.f);
    float2 uv = world_to_latlong_map(dir);
    result[threadId.x] = dot(normalize(dir), latlong_map_to_world(uv));
}

[numthreads(1024, 1, 1)]
void testSphericalCoordinatesRad(uint3 threadId: SV_DispatchThreadID)
{
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(0), threadId.x);
    float3 dir = normalize(sampleNext3D(sg) * 2.f - 1.f);
    float2 sph = cartesian_to_spherical_rad(dir);
    result[threadId.x] = dot(normalize(dir), spherical_to_cartesian_rad(sph));
}

[numthreads(25, 1, 1)]
void testErrorFunction(uint3 threadId: SV_DispatchThreadID)
{
    result[threadId.x] = erf(input[threadId.x]);
}

[numthreads(25, 1, 1)]
void testInverseErrorFunction(uint3 threadId: SV_DispatchThreadID)
{
    result[threadId.x] = erfinv(input[threadId.x]);
}
