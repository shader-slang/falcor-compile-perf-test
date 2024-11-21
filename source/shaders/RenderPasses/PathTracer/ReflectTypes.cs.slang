/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Dummy compute program for reflection of the structured buffer types.

     Falcor requires reflection data to be able to create a structured buffer,
     but the programs are not yet ready to be used when we create the buffers.
     Therefore we create this dummy program to make it easier.
 */
import RenderPasses.Shared.Denoising.NRDData;
import PathTracer;
import ColorType;
import GuideData;

ParameterBlock<PathTracer> pathTracer;

StructuredBuffer<ColorType> sampleColor;
StructuredBuffer<GuideData> sampleGuideData;
StructuredBuffer<NRDRadiance> sampleNRDRadiance;
StructuredBuffer<float> sampleNRDHitDist;
StructuredBuffer<float4> sampleNRDPrimaryHitNeeOnDelta;
StructuredBuffer<float4> sampleNRDEmission;
StructuredBuffer<float4> sampleNRDReflectance;

void main() {}
