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
 * This file contains functions for linearly-transformed cosines (LTCs).
 *
 * For details refer to:
 * "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines",
 * Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt, ACM SIGGRAPH 2016.
 *
 * Note: The code here isn't currently used and is largely untested.
 *
 * The LTC_Evaluate* functions use the approximate material properties in
 * BSDFProperties as a proxy for a simple spec-rough shading model,
 * including using the guide normal as normal since it includes normal mapping.
 */
import Scene.ShadingData;
import Scene.Lights.LightData;
import Utils.Math.MathHelpers;

struct LTCTerms
{
    float3x3 MinvS;
    float3x3 MS;
    float MdetS;

    float3x3 MinvD;
    float3x3 MD;
    float MdetD;

    float2 scaleBias;
};

struct LTCAreaLightParams
{
    SamplerState gLTCLookupSampler;
    Texture2D<float4> gLTCLookupTexture1;
    Texture2D<float4> gLTCLookupTexture2;
};

/**
 * Get LTC terms.
 * N = shading normal
 * V = vector to the eye from the point (outgoing direction)
 */
void LTC_GetTerms(float3 N, float3 V, float roughness, out LTCTerms terms, const LTCAreaLightParams params)
{
    float ndotv = saturate(dot(N, V));
    float2 uv = float2(roughness, sqrt(1.0f - ndotv));
    float4 t1 = params.gLTCLookupTexture1.SampleLevel(params.gLTCLookupSampler, uv, 0.f);
    float4 t2 = params.gLTCLookupTexture2.SampleLevel(params.gLTCLookupSampler, uv, 0.f);

    // Transposed compared to original implementation due to
    float3x3 Minv = float3x3(t1.x, 0.0f, t1.z, 0.0f, 1.0f, 0.0f, t1.y, 0.0f, t1.w);

    // construct orthonormal basis around N
    float3 T1, T2;
    T1 = normalize(V - N * dot(V, N));
    T2 = cross(N, T1);

    // World to tangent matrix
    float3x3 w2t = float3x3(T1, T2, N);

    terms.MinvS = mul(Minv, w2t);
    terms.MS = inverse(terms.MinvS);
    terms.MdetS = determinant(terms.MS);

    // diffuse
    terms.MinvD = w2t;
    terms.MD = transpose(terms.MinvD);
    terms.MdetD = 1.0f;

    terms.scaleBias = t2.xy;
}

void LTC_ClipQuadToHorizon(inout float3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0)
        config += 1;
    if (L[1].z > 0.0)
        config += 2;
    if (L[2].z > 0.0)
        config += 4;
    if (L[3].z > 0.0)
        config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }

    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}

float3 LTC_IntegrateEdgeV3(float3 v1, float3 v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;

    float theta_sintheta = (x > 0.0) ? v : 0.5 * rsqrt(max(1.0 - x * x, 1e-7)) - v;

    return cross(v1, v2) * theta_sintheta;
}

/**
 * P = point being shaded
 * Minv = LTC inverse matrix (see Heitz et al. 2016)
 * points vertices of the light
 */
float LTC_Evaluate(float3 P, float3x3 Minv, float3 points[4])
{
    // polygon (allocate 5 vertices for clipping)
    float3 L[5];
    L[0] = mul(Minv, (points[0] - P)); // TODO: What's the right order?!
    L[1] = mul(Minv, (points[1] - P));
    L[2] = mul(Minv, (points[2] - P));
    L[3] = mul(Minv, (points[3] - P));

    // integrate
    float sum;

    {
        int n = 0;
        LTC_ClipQuadToHorizon(L, n);

        if (n == 0)
            return 0.0;

        // project onto sphere
        L[0] = normalize(L[0]);
        L[1] = normalize(L[1]);
        L[2] = normalize(L[2]);
        L[3] = normalize(L[3]);
        L[4] = normalize(L[4]);

        float3 vsum = 0;

        // integrate
        vsum = LTC_IntegrateEdgeV3(L[0], L[1]);
        vsum += LTC_IntegrateEdgeV3(L[1], L[2]);
        vsum += LTC_IntegrateEdgeV3(L[2], L[3]);
        if (n >= 4)
            vsum += LTC_IntegrateEdgeV3(L[3], L[4]);
        if (n == 5)
            vsum += LTC_IntegrateEdgeV3(L[4], L[0]);

        sum = max(0.0, vsum.z);
    }

    return sum;
}

void LTC_GetLightVertices(const LightData Light, out float3 lightVertices[4])
{
    // TODO: Seems the code is written for clockwise winding, which is a bit unusual
    lightVertices[0] = mul(float4(-1.0f, -1.0f, 0.0f, 1.0f), Light.transMat).xyz;
    lightVertices[1] = mul(float4(-1.0f, 1.0f, 0.0f, 1.0f), Light.transMat).xyz;
    lightVertices[2] = mul(float4(1.0f, 1.0f, 0.0f, 1.0f), Light.transMat).xyz;
    lightVertices[3] = mul(float4(1.0f, -1.0f, 0.0f, 1.0f), Light.transMat).xyz;
}

float3 LTC_EvaluateDiffuse(
    const ShadingData sd,
    const BSDFProperties bsdfProperties,
    const LightData Light,
    const LTCAreaLightParams params
)
{
    float3 lightVertices[4];
    LTCTerms ltcTerms;

    LTC_GetTerms(bsdfProperties.guideNormal, sd.V, bsdfProperties.roughness, ltcTerms, params);
    LTC_GetLightVertices(Light, lightVertices);

    float diffLobe = LTC_Evaluate(sd.posW, ltcTerms.MinvD, lightVertices);
    float3 diffuse = bsdfProperties.diffuseReflectionAlbedo * diffLobe;

    return diffuse * Light.intensity;
}

float3 LTC_EvaluateSpecular(
    const ShadingData sd,
    const BSDFProperties bsdfProperties,
    const LightData Light,
    const LTCAreaLightParams params
)
{
    float3 lightVertices[4];
    LTCTerms ltcTerms;

    LTC_GetTerms(bsdfProperties.guideNormal, sd.V, bsdfProperties.roughness, ltcTerms, params);
    LTC_GetLightVertices(Light, lightVertices);

    float3 specColor =
        bsdfProperties.specularReflectance * ltcTerms.scaleBias.x + (1.0f - bsdfProperties.specularReflectance) * ltcTerms.scaleBias.y;
    float specLobe = LTC_Evaluate(sd.posW, ltcTerms.MinvS, lightVertices);
    float3 specular = specColor * specLobe;

    return specular * Light.intensity;
}

float3 LTC_Evaluate(const ShadingData sd, const BSDFProperties bsdfProperties, const LightData Light, const LTCAreaLightParams params)
{
    float3 lightVertices[4];
    LTCTerms ltcTerms;

    LTC_GetTerms(bsdfProperties.guideNormal, sd.V, bsdfProperties.roughness, ltcTerms, params);
    LTC_GetLightVertices(Light, lightVertices);

    float3 specColor =
        bsdfProperties.specularReflectance * ltcTerms.scaleBias.x + (1.0f - bsdfProperties.specularReflectance) * ltcTerms.scaleBias.y;
    float specLobe = LTC_Evaluate(sd.posW, ltcTerms.MinvS, lightVertices);
    float3 specular = specColor * specLobe;

    float diffLobe = LTC_Evaluate(sd.posW, ltcTerms.MinvD, lightVertices);
    float3 diffuse = bsdfProperties.diffuseReflectionAlbedo * diffLobe;

    return (diffuse + specular) * Light.intensity;
}
