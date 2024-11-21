/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import NRDData;

#if !defined(OUTPUT_NRD_DATA) || !defined(OUTPUT_NRD_ADDITIONAL_DATA)
#error NRDBuffers.slang is missing defines
#endif

static const bool kOutputNRDData = OUTPUT_NRD_DATA;
static const bool kOutputNRDAdditionalData = OUTPUT_NRD_ADDITIONAL_DATA;

struct NRDBuffers
{
    /// Output per-sample NRD radiance data. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<NRDRadiance> sampleRadiance;
    /// Output per-sample NEE on paths classified as delta, from delta+diffuse materials. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<float4> samplePrimaryHitNEEOnDelta;

    /// Output per-sample NRD hit distance data. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<float> sampleHitDist;
    /// Output per-sample NRD emission data. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<float4> sampleEmission;
    /// Output per-sample NRD reflectance data. Only valid if kOutputNRDData == true.
    RWStructuredBuffer<float4> sampleReflectance;

    /// Output per-pixel primary hit emission. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> primaryHitEmission;
    /// Output per-pixel primary hit diffuse reflectance. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> primaryHitDiffuseReflectance;
    /// Output per-pixel primary hit specular reflectance. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> primaryHitSpecularReflectance;

    /// Output per-pixel delta reflection reflectance. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaReflectionReflectance;
    /// Output per-pixel delta reflection emission. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaReflectionEmission;
    /// Output per-pixel delta reflection world normal, roughness, and material ID. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaReflectionNormWRoughMaterialID;

    /// Output per-pixel delta reflection path length. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float> deltaReflectionPathLength;
    /// Output per-pixel delta reflection hit distance. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float> deltaReflectionHitDist;

    /// Output per-pixel delta transmission reflectance. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaTransmissionReflectance;
    /// Output per-pixel delta transmission emission. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaTransmissionEmission;
    /// Output per-pixel delta transmission world normal, roughness, and material ID. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaTransmissionNormWRoughMaterialID;
    /// Output per-pixel delta transmission path length. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float> deltaTransmissionPathLength;
    /// Output per-pixel delta transmission world position. Only valid if kOutputNRDAdditionalData == true.
    RWTexture2D<float4> deltaTransmissionPosW;
};
