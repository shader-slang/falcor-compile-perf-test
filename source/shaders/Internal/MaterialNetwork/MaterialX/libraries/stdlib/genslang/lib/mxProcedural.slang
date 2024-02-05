/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

// On licenses:
// The code is mostly taken from the original MaterialX implementation for GLSL and OSL,
// with minimal modifications (e.g., names do not necessarily include the input type, since slang has overloading)
// The original license is a standard Apache 2.0 license, found here:
// https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/LICENSE The original files from which the code was taken had no license
// information whatsoever, so there was nothing to preserve. The standard license header found in many of the MaterialX source files is:
//
// TM & (c) 2017 Lucasfilm Entertainment Company Ltd. and Lucasfilm Ltd.
// All rights reserved.  See LICENSE.txt for license.
//

// Importing standard functions, taken from GLSL MaterialX implementation
__exported import Internal.MaterialNetwork.MaterialX.libraries.stdlib.genslang.lib.mxNoise;
__exported import Internal.MaterialNetwork.MaterialX.libraries.stdlib.genslang.lib.mxHsv;
// Importing BSDFs that the MaterialX standard PBR library expects to be present
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxBeer;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxBurley;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxConductor;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxDielectric;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxDielectricPlate;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxGeneralizedSchlick;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxOrenNayar;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxScratchConductor;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxSheen;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxSubsurface;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxThinFilm;
__exported import Internal.MaterialNetwork.MaterialX.MxBsdfs.MxTranslucent;
// Importing things needed by the MaterialX texture access nodes
__exported import Scene.Material.TextureHandle;
__exported import Scene.Material.TextureSampler;
// Importing volume phase functions
__exported import Rendering.Volumes.PhaseFunction;

// Enums (ints necessary due to codegen behavior) for the "model", "object", and "world" space keywords
static const int mx_space_model = 0;
static const int mx_space_object = 1;
static const int mx_space_world = 2;

// Enums (ints necessary due to codegen behavior) for the "tangent", and "object" normalmap space keywords
static const int mx_normalmap_space_tangent = 0;
static const int mx_normalmap_space_object = 0;

float3 mxBendNormal(float3 Norg, float3 Nmapped, float3 viewDir)
{
    float3 Ng = Norg;    // original normal
    float3 Ns = Nmapped; // normal mapped normal

    if (dot(Ng, viewDir) < 0.f)
    {
        Ns = -Nmapped;
        Ng = -Ng;
    }

    // Algorithm from Appendix A.3 of https://arxiv.org/abs/1705.01263
    float3 R = reflect(-viewDir, Ns);
    float a = dot(Ng, R);
    if (a >= 0)
        return Ns;

    // Otherwise perturb normal
    float b = max(0.001, dot(Ns, Ng));
    return normalize(viewDir + normalize(R - Ns * a / b));
}

// clang-format turned off to keep the 1:1 match with the original MaterialX files.
// clang-format off

#define M_FLOAT_EPS 1e-8

// MaterialX functions taken from the various .inline files that were too long to fit into "sourcecode" snippets directly in the .mtlx definition

// normalmap
float3 mx_normalmap(float3 value, int map_space, float normal_scale, float3 N, float3 T)
{
    // Decode the normal map.
    value = select(value == float3(0.0f), float3(0.0, 0.0, 1.0), value * 2.0 - 1.0);

    // Transform from tangent space if needed.
    if (map_space == mx_normalmap_space_tangent)
    {
        float3 B = normalize(cross(N, T));
        value.xy *= normal_scale;
        value = T * value.x + B * value.y + N * value.z;
    }

    // Normalize the result.
    return normalize(value);
}

// noise2d
float mx_noise_float(float2 texcoord) { return mx_perlin_noise_float(texcoord); }
float2 mx_noise_float2(float2 texcoord) { return mx_perlin_noise_vec3(texcoord).xy; }
float3 mx_noise_float3(float2 texcoord) { return mx_perlin_noise_vec3(texcoord); }
float4 mx_noise_float4(float2 texcoord) { return float4(mx_noise_float3(texcoord), mx_noise_float(texcoord + float2(19, 73))); }

// noise3d
float mx_noise_float(float3 position) { return mx_perlin_noise_float(position); }
float2 mx_noise_float2(float3 position) { return mx_perlin_noise_vec3(position).xy; }
float3 mx_noise_float3(float3 position) { return mx_perlin_noise_vec3(position); }
float4 mx_noise_float4(float3 position) { return float4(mx_noise_float3(position), mx_noise_float(position + float3(19, 73, 29))); }

// fractal3d
float2 mx_fractal_noise_float2(float3 position, int octaves, float lacunarity, float diminish) { return mx_fractal_noise_vec2(position, octaves, lacunarity, diminish); }
float3 mx_fractal_noise_float3(float3 position, int octaves, float lacunarity, float diminish) { return mx_fractal_noise_vec3(position, octaves, lacunarity, diminish); }
float4 mx_fractal_noise_float4(float3 position, int octaves, float lacunarity, float diminish) { return mx_fractal_noise_vec4(position, octaves, lacunarity, diminish); }

// modulo
float mx_modulo(float x, float y) { return x - y * floor(x / y); }
float2 mx_modulo(float2 x, float2 y) { return x - y * floor(x / y); }
float3 mx_modulo(float3 x, float3 y) { return x - y * floor(x / y); }
float4 mx_modulo(float4 x, float4 y) { return x - y * floor(x / y); }

// transform
float3 mx_transform_point(float3 p, int fromSpace, int toSpace, float4x4 objectFromWorld, float4x4 worldFromObject)
{
    if ((fromSpace == mx_space_model || fromSpace == mx_space_object) && toSpace == mx_space_world)
    {
        float4 hp = mul(worldFromObject, float4(p, 1));
        return hp.xyz / hp.w;
    }
    if (fromSpace == mx_space_world && (toSpace == mx_space_model || toSpace == mx_space_object))
    {
        float4 hp = mul(objectFromWorld, float4(p, 1));
        return hp.xyz / hp.w;
    }
    return p;
}

float3 mx_transform_vector(float3 p, int fromSpace, int toSpace, float4x4 objectFromWorld, float4x4 worldFromObject)
{
    if ((fromSpace == mx_space_model || fromSpace == mx_space_object) && toSpace == mx_space_world)
        return mul(worldFromObject, float4(p, 0)).xyz;
    if (fromSpace == mx_space_world && (toSpace == mx_space_model || toSpace == mx_space_object))
        return mul(objectFromWorld, float4(p, 0)).xyz;
    return p;
}

float3 mx_transform_normal(float3 p, int fromSpace, int toSpace, float3x3 objectFromWorldIT, float3x3 worldFromObjectIT)
{
    if ((fromSpace == mx_space_model || fromSpace == mx_space_object) && toSpace == mx_space_world)
        return mul(worldFromObjectIT, p);
    if (fromSpace == mx_space_world && (toSpace == mx_space_model || toSpace == mx_space_object))
        return mul(objectFromWorldIT, p);
    return p;
}

// rotate
float2 mx_rotate(float2 _in, float amount)
{
    float rotationRadians = radians(amount);
    float sa = sin(rotationRadians);
    float ca = cos(rotationRadians);
    return float2(ca * _in.x + sa * _in.y, -sa * _in.x + ca * _in.y);
}

float4x4 mx_rotationMatrix(float3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    return float4x4(
                oc * x * x + c,     oc * x * y - z * s, oc * z * x + y * s, 0.0,
                oc * x * y + z * s, oc * y * y + c,     oc * y * z - x * s, 0.0,
                oc * z * x - y * s, oc * y * z + x * s, oc * z * z + c,     0.0,
                0.0,                0.0,                0.0,                1.0);
}

float3 mx_rotate(float3 _in, float amount, float3 axis)
{
    float rotationRadians = radians(amount);
    float4x4 m = mx_rotationMatrix(axis, rotationRadians);
    return mul(m, float4(_in, 1.0)).xyz;
}

float mx_burn(float fg, float bg, float mixval)
{
    if (abs(fg) < M_FLOAT_EPS)
        return 0;
    return mixval * (1.f - ((1.f - bg) / fg)) + ((1.f - mixval) * bg);
}

float3 mx_burn(float3 fg, float3 bg, float mixval)
{
    return float3(
        mx_burn(fg.x, bg.x, mixval),
        mx_burn(fg.y, bg.y, mixval),
        mx_burn(fg.z, bg.z, mixval)
    );
}

float4 mx_burn(float4 fg, float4 bg, float mixval)
{
    return float4(
        mx_burn(fg.x, bg.x, mixval),
        mx_burn(fg.y, bg.y, mixval),
        mx_burn(fg.z, bg.z, mixval),
        mx_burn(fg.w, bg.w, mixval)
    );
}

float mx_dodge(float fg, float bg, float mixval)
{
    if (abs(1.0 - fg) < M_FLOAT_EPS)
        return 0;
    return mixval * (bg / (1.0 - fg)) + ((1.0 - mixval) * bg);
}

float3 mx_dodge(float3 fg, float3 bg, float mixval)
{
    return float3(
        mx_dodge(fg.x, bg.x, mixval),
        mx_dodge(fg.y, bg.y, mixval),
        mx_dodge(fg.z, bg.z, mixval)
    );
}

float4 mx_dodge(float4 fg, float4 bg, float mixval)
{
    return float4(
        mx_dodge(fg.x, bg.x, mixval),
        mx_dodge(fg.y, bg.y, mixval),
        mx_dodge(fg.z, bg.z, mixval),
        mx_dodge(fg.w, bg.w, mixval)
    );
}

float mx_overlay(float fg, float bg)
{
    return (fg < 0.5) ? (2.0 * fg * bg) : (1.0 - (1.0 - fg) * (1.0 - bg));
}

float2 mx_overlay(float2 fg, float2 bg)
{
    return vec2(mx_overlay(fg.x, bg.x),
                mx_overlay(fg.y, bg.y));
}

float3 mx_overlay(float3 fg, float3 bg)
{
    return vec3(mx_overlay(fg.x, bg.x),
                mx_overlay(fg.y, bg.y),
                mx_overlay(fg.z, bg.z));
}

float4 mx_overlay(float4 fg, float4 bg)
{
    return vec4(mx_overlay(fg.x, bg.x),
                mx_overlay(fg.y, bg.y),
                mx_overlay(fg.z, bg.z),
                mx_overlay(fg.w, bg.w));
}

float4 mx_disjointover(float4 fg, float4 bg, float mixval)
{
    float4 result;
    float summedAlpha = fg.w + bg.w;

    if (summedAlpha <= 1.0)
    {
        result.xyz = fg.xyz + bg.xyz;
    }
    else
    {
        if (abs(bg.w) < M_FLOAT_EPS)
        {
            result.xyz = vec3(0.0);
        }
        else
        {
            float x = (1.0 - fg.w) / bg.w;
            result.xyz = fg.xyz + bg.xyz * x;
        }
    }
    result.w = min(summedAlpha, 1.0);

    result.xyz = result.xyz * mixval + (1.0 - mixval) * bg.xyz;
    result.w = result.w * mixval + (1.0 - mixval) * bg.w;

    return result;
}
