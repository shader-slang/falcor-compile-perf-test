/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "Internal/Scene/Materials/NeuralMaterialTypes.slangh"
#include "Internal/Scene/Materials/NeuralMaterialSamplingStrategies.slangh"

import Rendering.Materials.IMaterialInstance;
import Scene.Displacement.DisplacementMapping;
import RenderPasses.PythonPathTracer.HandleDisplacement;
import RenderPasses.PythonPathTracer.FilterHelpers;
import Scene.Material.TextureSampler;
import Scene.Scene;
import Scene.Shading;
import Scene.ShadingData;
import Utils.Math.MathHelpers;

static const uint32_t kNumExportedBsdfs = NUM_EXPORTED_BSDFS;

#if 0
// TODO: Move into MathHelpers.slang
void cartesian_to_rusinkiewicz(float3 wi, float3 wo, out float2 h_enc, out float3 d_enc)
{
    float3 half = normalize(wi + wo);
    float2 half_sph = cartesian_to_spherical_rad(half);
    float3 temp = rotate_z(wi, -half_sph.y);
    float3 d = rotate_y(temp, -half_sph.x);
    float2 d_sph = cartesian_to_spherical_rad(d);

    h_enc = half_sph;
    d_enc = float3(d_sph.x, sin(d_sph.y), cos(d_sph.y));
}

void rusinkiewicz_to_cartesian(float2 h_enc, float3 d_enc, out float3 wi, out float3 wo)
{
    float2 d_sph = float2(d_enc.x, atan2(d_enc.y, d_enc.z));
    float2 half_sph = h_enc;

    float3 d = spherical_to_cartesian_rad(d_sph);
    float3 temp = rotate_y(d, half_sph.x);
    wi = rotate_z(temp, half_sph.y);
    float3 half = spherical_to_cartesian_rad(half_sph);
    wo = normalize(2 * dot(wi, half) * half - wi);
}
#endif

/** Structs describing the geometry surrounding the material sample
 */
struct MaterialSampleGeometryInfo
{
    float3 cameraPos;
    StaticVertexData vertices[3];
    VertexData v;
    float lod;
};

/** Structs describing a material sample.
 */

struct MaterialSampleBasic
{
    __init(inout ShadingData sd, const MaterialSampleGeometryInfo info, float3 wi_, bool training)
    {
        valid = 0.0f;
        uv = sd.uv;
        wi = wi_;
        wo = float3(0.0f);
        pdf = 0.0f;
        weight = float3(0.0f);
    }
    [mutating]
    void setWo(float3 wo_) { wo = wo_; }
    [mutating]
    void setPdf(float pdf_) { pdf = pdf_; }
    [mutating]
    void setWeight(float3 weight_) { weight = weight_; }
    [mutating]
    void setMaterialProperties(const ShadingData sd, const IMaterialInstance mi, float lod) {}
    [mutating]
    void permuteChannels(float rand01) {}
    float2 getUV() { return uv; }
    float3 getWi() { return wi; }
    float3 getWo() { return wo; }
    float getPdf() { return pdf; }
    float3 getWeight() { return weight; }
    [mutating]
    void setFilterFootprint(float2 ellipseX, float2 ellipseY, float mipLevel) {}
    void getFilterFootprint(out float2 ellipseX, out float2 ellipseY) { ellipseX = ellipseY = float2(0.0f); }
    [mutating]
    void markValid() { valid = 1.0f; }

    float valid;   ///< 1 = valid sample, 0 = invalid.
    float2 uv;     ///< Texture coordinate.
    float3 wi;     ///< Incident direction.
    float3 wo;     ///< Outgoing direction (sampled).
    float pdf;     ///< pdf with respect to solid angle for the sampled direction (wo).
    float3 weight; ///< f(wi, wo) * dot(wo, n).
};

struct MaterialSampleWithShFrame
{
    __init(inout ShadingData sd, const MaterialSampleGeometryInfo info, float3 wi_, bool training)
    {
        let base = MaterialSampleBasic(sd, info, wi_, training);

        valid = base.valid;
        uv = base.uv;
        wi = base.wi;
        wo = base.wo;
        pdf = base.pdf;
        weight = base.weight;

        // Set normal and tangent to account only for differences between
        // local geometry frame and local shading frame.
        VertexData v = {};
        v.posW = float3(0, 0, 0);
        v.normalW = float3(0, 0, 1);
        v.tangentW = float4(1, 0, 0, 1);
        v.texC = uv;
        v.faceNormalW = v.normalW;
        v.curveRadius = 0.f;
        v.coneTexLODValue = 0.f;
        // Query the material system to prepare shading data.
        ShadingData sd2 = gScene.materials.prepareShadingData(v, sd.materialID, wi, ExplicitLodTextureSampler(info.lod));
        normal = sd2.frame.N;
        tangent = sd2.frame.T;
    }
    [mutating]
    void setWo(float3 wo_) { wo = wo_; }
    [mutating]
    void setPdf(float pdf_) { pdf = pdf_; }
    [mutating]
    void setWeight(float3 weight_) { weight = weight_; }
    [mutating]
    void setMaterialProperties(const ShadingData sd, const IMaterialInstance mi, float lod) {}
    [mutating]
    void permuteChannels(float rand01) {}
    float2 getUV() { return uv; }
    float3 getWi() { return wi; }
    float3 getWo() { return wo; }
    float getPdf() { return pdf; }
    float3 getWeight() { return weight; }
    [mutating]
    void setFilterFootprint(float2 ellipseX, float2 ellipseY, float mipLevel) {}
    void getFilterFootprint(out float2 ellipseX, out float2 ellipseY) { ellipseX = ellipseY = float2(0.0f); }
    [mutating]
    void markValid() { valid = 1.0f; }

    float valid;    ///< 1 = valid sample, 0 = invalid.
    float2 uv;      ///< Texture coordinate.
    float3 wi;      ///< Incident direction.
    float3 wo;      ///< Outgoing direction (sampled).
    float3 normal;  ///< Shading normal
    float3 tangent; ///< Shading tangent
    float pdf;      ///< pdf with respect to solid angle for the sampled direction (wo).
    float3 weight;  ///< f(wi, wo) * dot(wo, n).
};

struct MaterialSampleWithShFrameExtended
{
    __init(inout ShadingData sd, const MaterialSampleGeometryInfo info, float3 wi_, bool training)
    {
        let base = MaterialSampleWithShFrame(sd, info, wi_, training);

        valid = base.valid;
        uv = base.uv;
        wi = base.wi;
        wo = base.wo;
        pdf = base.pdf;
        weight = base.weight;
        // Set default values that will be overwritten by setMaterialProperties
        for (uint i = 0; i < kNumExportedBsdfs; i++)
        {
            normal[i] = float3(0, 0, 1);
            tangent[i] = float3(1, 0, 0);
            roughness[i] = float2(0.5f);
            albedo[i] = float3(0.f);
            bsdfWeight[i] = 0.f;
            scratch[i] = float3(0.f);
            bsdfIndex[i] = 0.f;
        }

        float hitT = length(info.cameraPos - sd.posW);
        computeFilterEllipse(info.v, info.vertices, -sd.V, hitT, ellipseX, ellipseY);

        level = -1.0f;
    }
    [mutating]
    void setWo(float3 wo_) { wo = wo_; }
    [mutating]
    void setPdf(float pdf_) { pdf = pdf_; }
    [mutating]
    void setWeight(float3 weight_) { weight = weight_; }
    [mutating]
    void setMaterialProperties(const ShadingData sd, const IMaterialInstance mi, float lodLevel)
    {
        // Set normal and tangent to account only for differences between
        // local geometry frame and local shading frame.
        // This code is needed for calls that come from the python path tracer.
        // The material data generator already operates in [0,0,1] frame.
        VertexData v = {};
        v.posW = float3(0, 0, 0);
        v.normalW = float3(0, 0, 1);
        v.tangentW = float4(1, 0, 0, 1);
        v.texC = uv;
        v.faceNormalW = v.normalW;
        v.curveRadius = 0.f;
        v.coneTexLODValue = 0.f;
        // Query the material system to prepare shading data.
        ExplicitLodTextureSampler lod = { lodLevel };
        ShadingData sd2 = gScene.materials.prepareShadingData(v, sd.materialID, float3(0, 0, 1), lod);
        IMaterialInstance mi2 = gScene.materials.getMaterialInstance(sd2, lod);

        // This expects wo to be set correctly (i.e. setWo() must be called before this one).
        // BSDFProperties props = mi2.getProperties(sd2);
        ExtraBSDFProperties extraProps = mi2.getExtraBSDFProperties(sd2, wo);

        // // Guide normal + other aggregate information
        // normal[0] = props.guideNormal;
        // tangent[0] = float3(0, 0, 0);
        // roughness[0] = props.roughness;
        // albedo[0] = props.diffuseReflectionAlbedo + props.diffuseTransmissionAlbedo;
        // bsdfWeight[0] = 0.f; // empty

        // Write the first `kNumExportedBsdfs` BSDFs.
        uint i = 0;
        for (; i < min(extraProps.bsdfCount, kNumExportedBsdfs); i++)
        {
            normal[i] = extraProps.bsdfN[i];
            tangent[i] = extraProps.bsdfT[i];
            albedo[i] = extraProps.bsdfAlbedo[i];
            roughness[i] = extraProps.bsdfRoughness[i];
            bsdfWeight[i] = extraProps.bsdfWeight[i];
            scratch[i] = extraProps.bsdfScratch[i];
            bsdfIndex[i] = extraProps.bsdfIndex[i];
        }

        // The parameters of the the remaining BSDFs are aggregated to the last BSDF.
        // It doesn't make a whole lot of sense, but it's probably better than ignoring them.
        uint count = 1;
        for (; i < extraProps.bsdfCount; i++)
        {
            normal[kNumExportedBsdfs - 1] += extraProps.bsdfN[i];
            tangent[kNumExportedBsdfs - 1] += extraProps.bsdfT[i];
            albedo[kNumExportedBsdfs - 1] += extraProps.bsdfAlbedo[i];
            roughness[kNumExportedBsdfs - 1] += extraProps.bsdfRoughness[i];
            bsdfWeight[kNumExportedBsdfs - 1] += extraProps.bsdfWeight[i];
            scratch[kNumExportedBsdfs - 1] = extraProps.bsdfScratch[i];
            bsdfIndex[kNumExportedBsdfs - 1] = extraProps.bsdfIndex[i];
            count++;
        }
        float norm = 1.f / count;
        normal[kNumExportedBsdfs - 1] *= norm;
        tangent[kNumExportedBsdfs - 1] *= norm;
        albedo[kNumExportedBsdfs - 1] *= norm;
        roughness[kNumExportedBsdfs - 1] *= norm;
        bsdfWeight[kNumExportedBsdfs - 1] *= norm;
        scratch[kNumExportedBsdfs - 1] *= norm;
        bsdfIndex[kNumExportedBsdfs - 1] *= norm;
    }

    [mutating]
    void permuteChannels(float rand01)
    {
        if (rand01 < 1.f / 6.f)
            return;                  // 0,1,2
        else if (rand01 < 2.f / 6.f) // (0, 2, 1)
        {
            for (uint i = 0; i < kNumExportedBsdfs; i++)
            {
                albedo[i] = float3(albedo[i].x, albedo[i].z, albedo[i].y);
                bsdfWeight[i] = float3(bsdfWeight[i].x, bsdfWeight[i].z, bsdfWeight[i].y);
            }
            weight = float3(weight.x, weight.z, weight.y);
        }
        else if (rand01 < 3.f / 6.f) // (1, 0, 2)
        {
            for (uint i = 0; i < kNumExportedBsdfs; i++)
            {
                albedo[i] = float3(albedo[i].y, albedo[i].x, albedo[i].z);
                bsdfWeight[i] = float3(bsdfWeight[i].y, bsdfWeight[i].x, bsdfWeight[i].z);
            }
            weight = float3(weight.y, weight.x, weight.z);
        }
        else if (rand01 < 4.f / 6.f) // (1, 2, 0)
        {
            for (uint i = 0; i < kNumExportedBsdfs; i++)
            {
                albedo[i] = float3(albedo[i].y, albedo[i].z, albedo[i].x);
                bsdfWeight[i] = float3(bsdfWeight[i].y, bsdfWeight[i].z, bsdfWeight[i].x);
            }
            weight = float3(weight.y, weight.z, weight.x);
        }
        else if (rand01 < 5.f / 6.f) // (2, 0, 1)
        {
            for (uint i = 0; i < kNumExportedBsdfs; i++)
            {
                albedo[i] = float3(albedo[i].z, albedo[i].x, albedo[i].y);
                bsdfWeight[i] = float3(bsdfWeight[i].z, bsdfWeight[i].x, bsdfWeight[i].y);
            }
            weight = float3(weight.z, weight.x, weight.y);
        }
        else // (2, 1, 0)
        {
            for (uint i = 0; i < kNumExportedBsdfs; i++)
            {
                albedo[i] = float3(albedo[i].z, albedo[i].y, albedo[i].x);
                bsdfWeight[i] = float3(bsdfWeight[i].z, bsdfWeight[i].y, bsdfWeight[i].x);
            }
            weight = float3(weight.z, weight.y, weight.x);
        }
    }
    float2 getUV() { return uv; }
    float3 getWi() { return wi; }
    float3 getWo() { return wo; }
    float getPdf() { return pdf; }
    float3 getWeight() { return weight; }
    [mutating]
    void setFilterFootprint(float2 ellipseX, float2 ellipseY, float mipLevel)
    {
        this.ellipseX = ellipseX;
        this.ellipseY = ellipseY;
        this.level = mipLevel;
    }
    void getFilterFootprint(out float2 ellipseX, out float2 ellipseY)
    {
        ellipseX = this.ellipseX;
        ellipseY = this.ellipseY;
    }
    [mutating]
    void markValid() { valid = 1.0f; }

    float valid; ///< 1 = valid sample, 0 = invalid.
    float2 uv;   ///< Texture coordinate.
    float3 wi;   ///< Incident direction.
    float3 wo;   ///< Outgoing direction (sampled).
    // The following parameters are stored in arrays,
    // where the last entry aggregates all BSDFs that exceed the `kNumExportedBsdfs` limit.

    float3 normal[kNumExportedBsdfs];
    float3 tangent[kNumExportedBsdfs];
    float3 albedo[kNumExportedBsdfs];
    float2 roughness[kNumExportedBsdfs];
    float3 bsdfWeight[kNumExportedBsdfs];
    float level;
    float3 scratch[kNumExportedBsdfs];
    float bsdfIndex[kNumExportedBsdfs];
    // float3 specular; /// Currently not used as material X does not feature it.

    float pdf;     ///< pdf with respect to solid angle for the sampled direction (wo).
    float3 weight; ///< f(wi, wo) * dot(wo, n).

    float2 ellipseX;
    float2 ellipseY;
};
struct MaterialSampleDisplacement
{
    float2 uvHeightDerivative(float3 dir)
    {
        float3 N = normalize(cross(dP1, dP2));
        float3x3 M = {
            dP1.x,
            dP2.x,
            N.x,
            dP1.y,
            dP2.y,
            N.y,
            dP1.z,
            dP2.z,
            N.z,
        };
        float3 dUVWdP = mul(inverse(M), dir);
        float2 dSTdP = dT1 * dUVWdP.x + dT2 * dUVWdP.y;

        return dSTdP / dUVWdP.z;
    }

    __init(inout ShadingData sd, const MaterialSampleGeometryInfo info, float3 wi_, bool training)
    {
        let vertices = info.vertices;
        dP1 = vertices[1].position - vertices[0].position;
        dP2 = vertices[2].position - vertices[0].position;
        dT1 = vertices[1].texCrd - vertices[0].texCrd;
        dT2 = vertices[2].texCrd - vertices[0].texCrd;

        float2 dUVdH = uvHeightDerivative(-sd.V);

        DisplacementContext dc = makeDisplacementContext(sd, vertices);

        uv2 = float2(0.0f);
        float h = 0.0f;
        float h0 = dc.hTop;
        float h1 = dc.hBot;

        uv0 = sd.uv;
        uv1 = sd.uv + dUVdH * (h1 - h);

        if (training)
        {
            dc.primaryHit(sd, -sd.V);
            displacement = (dc.curHeight - dc.hBot) / (dc.hTop - dc.hBot);
        }
        else
        {
            ShadingData tmp = sd;
            dc.primaryHit(tmp, -sd.V);
            displacement = (dc.curHeight - dc.hBot) / (dc.hTop - dc.hBot);

            dP1 = sd.frame.toLocal(dP1);
            dP2 = sd.frame.toLocal(dP2);
        }

        let base = MaterialSampleBasic(sd, info, wi_, training);

        valid = base.valid;
        uv = base.uv;
        wi = base.wi;
        wo = base.wo;
        pdf = base.pdf;
        weight = base.weight;
    }

    [mutating]
    void setWo(float3 wo_)
    {
        wo = wo_;
        float2 dUVdH = uvHeightDerivative(wo);
        uv2 = uv1 - dUVdH * 0.1f; // TODO hardcoded
    }
    [mutating]
    void setPdf(float pdf_) { pdf = pdf_; }
    [mutating]
    void setWeight(float3 weight_) { weight = weight_; }
    [mutating]
    void setMaterialProperties(const ShadingData sd, const IMaterialInstance mi, float lod) {}
    [mutating]
    void permuteChannels(float rand01) {}
    float2 getUV() { return uv; }
    float3 getWi() { return wi; }
    float3 getWo() { return wo; }
    float getPdf() { return pdf; }
    float3 getWeight() { return weight; }
    [mutating]
    void setFilterFootprint(float2 ellipseX, float2 ellipseY, float mipLevel) {}
    void getFilterFootprint(out float2 ellipseX, out float2 ellipseY) { ellipseX = ellipseY = float2(0.0f); }
    [mutating]
    void markValid() { valid = 1.0f; }

    float valid;   ///< 1 = valid sample, 0 = invalid.
    float2 uv;     ///< Texture coordinate.
    float3 wi;     ///< Incident direction.
    float3 wo;     ///< Outgoing direction (sampled).
    float pdf;     ///< pdf with respect to solid angle for the sampled direction (wo).
    float3 weight; ///< f(wi, wo) * dot(wo, n).
    float2 uv0, uv1, uv2;
    float3 dP1, dP2;
    float2 dT1, dT2;
    float displacement;
};

#if NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_UNKNOWN
#error Define NEURAL_MATERIAL_TYPE cannot be set to NEURAL_MATERIAL_TYPE_UNKNOWN
#endif
#if NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_BASIC || NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_LATENT_TEXTURES
typedef MaterialSampleBasic MaterialSample;
#elif NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_WITH_SHFRAME
typedef MaterialSampleWithShFrame MaterialSample;
#elif NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_WITH_SHFRAME_EXTENDED
typedef MaterialSampleWithShFrameExtended MaterialSample;
#elif NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_DISPLACEMENT
typedef MaterialSampleDisplacement MaterialSample;
#endif
