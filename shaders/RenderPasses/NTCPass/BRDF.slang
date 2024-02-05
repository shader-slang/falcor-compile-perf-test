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

inline float clamp(float x, float x_min, float x_max)
{
    return max(min(x, x_max), x_min);
}

inline float sqr(float x)
{
    return x * x;
}

inline float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m;
}

inline float GTR1(float NdotH, float a)
{
    if (a >= 1)
    {
        return 1 / M_PI;
    }
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (M_PI * log(a2) * t);
}

inline float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (M_PI * t * t);
}

inline float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1 / (M_PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
}

inline float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

inline float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
}

inline float3 BRDF(float3 L, float3 V, float3 N, float3 baseColor, float roughness = 0.25, float metallic = 0.0, float specular = 1.0)
{
    const float subsurface = 0;
    const float specularTint = 0;
    const float anisotropic = 0;
    const float sheen = 0;
    const float sheenTint = .5;
    const float clearcoat = 0;
    const float clearcoatGloss = 1;

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL < 0 || NdotV < 0)
        return float3(0);

    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    float3 Cdlin = baseColor;
    float Cdlum = .3f * Cdlin[0] + .6f * Cdlin[1] + .1f * Cdlin[2]; // Luminance approximation.

    float3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : float3(1); // Normalize luminance to isolate hue and saturation.
    float3 Cspec0 = lerp(specular * float3(.08f) * lerp(float3(1.f), Ctint, specularTint), Cdlin, metallic);
    float3 Csheen = lerp(float3(1), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and lerp in diffuse retro-reflection based on roughness.
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5f + 2 * LdotH * LdotH * roughness;
    float Fd = lerp(1.f, Fd90, FL) * lerp(1.f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness.
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.f, Fss90, FL) * lerp(1.f, Fss90, FV);
    float ss = 1.25f * (Fss * (1 / (NdotL + NdotV) - .5f) + .5f);

    // Specular.
    float a = sqr(roughness);
    float Ds = GTR2(NdotH, a);
    float FH = SchlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, float3(1), FH);
    float Gs;
    Gs = smithG_GGX(NdotL, a);
    Gs *= smithG_GGX(NdotV, a);

    // Sheen.
    float3 Fsheen = FH * sheen * Csheen;

    // Clearcoat (ior = 1.5 -> F0 = 0.04).
    float Dr = GTR1(NdotH, lerp(.1, .001, clearcoatGloss));
    float Fr = lerp(.04f, 1.f, FH);
    float Gr = smithG_GGX(NdotL, .25f) * smithG_GGX(NdotV, .25f);

    return ((1 / M_PI) * lerp(Fd, ss, subsurface) * Cdlin + Fsheen) * (1 - metallic) + Gs * Fs * Ds + .25f * clearcoat * Gr * Fr * Dr;
}
