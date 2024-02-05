/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import BsdfGGX;

// Output
RWTexture2D<float4> gOutputBwdDColor;

BSDFParameters getMaterialParams(uint materialID, uint2 pixel)
{
    BSDFParameters bsdfParams;
    if (materialID == 0)
    {
        bsdfParams.albedo = float3(0.7f, 0.7f, 0.7f);
        bsdfParams.roughness = 0.1f;
    }
    else if (materialID == 1)
    {
        bsdfParams.albedo = float3(0.9f, 0.6f, 0.4f);
        bsdfParams.roughness = 0.1f;
    }
    else
    {
        bsdfParams.albedo = float3(0.5f, 0.5f, 0.5f);
        bsdfParams.roughness = 0.5f;
    }
    return bsdfParams;
}

[TreatAsDifferentiable]
BSDFParameters.Differential getMaterialParamsDifferential(uint materialID, uint2 pixel)
{
    BSDFParameters.Differential result;
    result.albedo = float3(0.f);
    result.roughness = (materialID == 1 ? 1.f : 0.f);
    return result;
}

[ForwardDerivativeOf(getMaterialParams)]
[TreatAsDifferentiable]
DifferentialPair<BSDFParameters> __fwd_d_getDiffMaterialParams(uint materialID, uint2 pixel)
{
    return DifferentialPair<BSDFParameters>(
        getMaterialParams(materialID, pixel),
        getMaterialParamsDifferential(materialID, pixel));
}

[BackwardDerivativeOf(getMaterialParams)]
[TreatAsDifferentiable]
void __bwd_d_getDiffMaterialParams(uint materialID, uint2 pixel, BSDFParameters.Differential dOut)
{
    float4 result = float4(materialID == 1 ? dOut.roughness : 0.f, 0.f, 0.f, 0.f);
    gOutputBwdDColor[pixel] += result;
}
