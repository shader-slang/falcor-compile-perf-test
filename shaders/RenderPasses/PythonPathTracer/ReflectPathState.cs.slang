/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/**
 * Dummy compute program for reflection of the PathState struct.
 *
 * Falcor requires reflection data to be able to create a structured buffer,
 * but the programs are not yet ready to be used when we setup the path state.
 * Therefore we create this dummy program to make it easier.
 */
import Rendering.Materials.InteriorList;
import RenderPasses.LegacyPathTracer.WavefrontPathTracer.PathState;
import RenderPasses.LegacyPathTracer.PathTracer.RayFootprintData;
import Plugins.MaterialDataGenerator.MaterialDataShared;

StructuredBuffer<uint> pathFlags;
StructuredBuffer<PathState> pathState;
StructuredBuffer<PathRays> pathRays;
StructuredBuffer<float3> pathTransmission;
StructuredBuffer<InteriorList> pathInteriorList;
StructuredBuffer<float3> pathOutput;
StructuredBuffer<PackedHitInfo> pathHit;
StructuredBuffer<RayFootprint> pathRayFootprint;
StructuredBuffer<MaterialSample> materialSamples;

void main() {}
