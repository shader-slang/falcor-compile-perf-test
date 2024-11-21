/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
cbuffer BlitParamsCB
{
    float2 gOffset;
    float2 gScale;
    float4 gCompTransformR;
    float4 gCompTransformG;
    float4 gCompTransformB;
    float4 gCompTransformA;
}

#if SAMPLE_COUNT <= 1
Texture2D<float4> gTex;
#else
Texture2DMS<float4> gTex;
#endif

SamplerState gSampler;
SamplerState gSamplerR;
SamplerState gSamplerG;
SamplerState gSamplerB;
SamplerState gSamplerA;

struct VsOut
{
    float2 texC : TEXCOORD;
    float4 posH : SV_POSITION;
};

VsOut vsMain(float4 posS: POSITION, float2 texC: TEXCOORD)
{
    VsOut vOut;
    vOut.texC = texC * gScale + gOffset;
    vOut.posH = posS;
    return vOut;
}

#if DST_INT
uint4 psMain(float2 texC: TEXCOORD) : SV_TARGET
#else
float4 psMain(float2 texC: TEXCOORD) : SV_TARGET
#endif
{
#if SAMPLE_COUNT <= 1

#if COMPLEX_BLIT
#if SRC_INT
#error Complex blit not supported for integer source format
#else
    float4 srcR = gTex.Sample(gSamplerR, texC);
    float4 srcG = gTex.Sample(gSamplerG, texC);
    float4 srcB = gTex.Sample(gSamplerB, texC);
    float4 srcA = gTex.Sample(gSamplerA, texC);

    float4 res;
    res.r = dot(gCompTransformR, srcR);
    res.g = dot(gCompTransformG, srcG);
    res.b = dot(gCompTransformB, srcB);
    res.a = dot(gCompTransformA, srcA);

    return res;
#endif
#else // !COMPLEX_BLIT
#if SRC_INT
    uint4 res = asuint(gTex.Sample(gSampler, texC));
#else
    float4 res = gTex.Sample(gSampler, texC);
#endif
    return res;
#endif

#else // SAMPLE_COUNT > 1

#if COMPLEX_BLIT
#error Complex blit not supported for sample count > 1
#else // !COMPLEX_BLIT
#if SRC_INT
#error Blit for sample count > 1 not supported for integer source format
#else
    uint3 dims;
    gTex.GetDimensions(dims.x, dims.y, dims.z);
    uint2 crd = (uint2)(float2(dims.xy) * texC);
    float4 c = float4(0, 0, 0, 0);

    [unroll]
    for (uint i = 0; i < SAMPLE_COUNT; i++)
    {
        c += gTex.Load(crd, i);
    }

    c /= SAMPLE_COUNT;
    return c;
#endif
#endif

#endif
}
