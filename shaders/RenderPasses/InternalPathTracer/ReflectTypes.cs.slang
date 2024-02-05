/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Dummy compute program for reflection of the structured buffer types.

     Falcor requires reflection data to be able to create a structured buffer,
     but the programs are not yet ready to be used when we create the buffers.
     Therefore we create this dummy program to make it easier.
 */
import RenderPasses.Shared.Denoising.NRDData;
import NeuralRadianceCache.NeuralRadianceCache;
import InternalPathTracer;
import QueueData;
import ColorType;
import GuideData;

ParameterBlock<InternalPathTracer> pathTracer;

StructuredBuffer<PackedSpecularPath> generalQueue;
StructuredBuffer<ColorType> sampleColor;
StructuredBuffer<GuideData> sampleGuideData;
StructuredBuffer<NRDRadiance> sampleNRDRadiance;
StructuredBuffer<float> sampleNRDHitDist;
StructuredBuffer<float4> sampleNRDPrimaryHitNeeOnDelta;
StructuredBuffer<float4> sampleNRDEmission;
StructuredBuffer<float4> sampleNRDReflectance;
StructuredBuffer<NRC::PathInfo> pathInfo;
StructuredBuffer<NRC::PackedPathVertex> pathVertices;
StructuredBuffer<NRC::RadianceParams> radianceParams;

void main() {}
