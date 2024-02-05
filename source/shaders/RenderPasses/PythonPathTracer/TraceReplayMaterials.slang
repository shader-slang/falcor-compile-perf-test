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
__exported import RenderPasses.PythonPathTracer.PythonRayState;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.MathHelpers;
import Scene.Scene;
import Scene.ShadingData;
import Scene.SceneTypes;
import Scene.Shading;

cbuffer PythonCB
{
    PythonRayState gPythonSample;
    PythonRayState gPythonInference;
    Buffer<uint> gIsPythonMaterial;
    bool gHavePythonEval;
    bool gHavePythonSampling;
    bool gForceCosineSampling;
    bool gUseNormalMap;
    bool gFurnaceTest;
};

enum TracingMode
{
    TraceEvalCalls,
    TraceSampleOrPdfCalls,
};

struct TracingMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    IMaterialInstance baseMaterial;
    uint writeIdx;
    TracingMode mode;
    float lod;

    __init(IMaterialInstance mi, MaterialSample ms, TracingMode mode, MaterialSampleGeometryInfo info, uint idx)
    {
        baseMaterial = mi;
        writeIdx = idx;
        this.mode = mode;
        this.lod = info.lod;

        if (mode == TracingMode::TraceEvalCalls)
        {
            gPythonInference.samples[writeIdx] = ms;
        }
        else
        {
            gPythonSample.samples[writeIdx] = ms;
        }
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        if (mode != TracingMode::TraceEvalCalls)
            return float3(1.0f); // This should not happen

        gPythonInference.samples[writeIdx].setWo(sd.frame.toLocal(wo));
        gPythonInference.samples[writeIdx].setMaterialProperties(sd, baseMaterial, lod);
        gPythonInference.samples[writeIdx].markValid();
        gPythonInference.markActive(writeIdx);

        return float3(1.0f);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        if (mode != TracingMode::TraceSampleOrPdfCalls)
            return false; // This should not happen

        gPythonSample.samples[writeIdx].setMaterialProperties(sd, baseMaterial, lod);

        if (gForceCosineSampling)
        {
            gPythonSample.samples[writeIdx].setWo(sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf));
        }
        else if (!gHavePythonSampling)
        {
            if (!baseMaterial.sample(sd, sg, result, useImportanceSampling))
                return false;
            gPythonSample.samples[writeIdx].setWo(sd.frame.toLocal(result.wo));
        }

        gPythonSample.samples[writeIdx].markValid();
        gPythonSample.markActive(writeIdx);

        return false;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        if (mode != TracingMode::TraceSampleOrPdfCalls)
            return 0.0f; // This should not happen

        gPythonInference.samples[writeIdx].setWo(sd.frame.toLocal(wo));
        gPythonInference.samples[writeIdx].setMaterialProperties(sd, baseMaterial, lod);
        gPythonInference.samples[writeIdx].markValid();
        gPythonInference.markActive(writeIdx);

        return 0.0f;
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        return p;
    }

    uint getLobeTypes(const ShadingData sd) { return uint(LobeType::Specular); }

    ExtraBSDFProperties getExtraBSDFProperties(const ShadingData sd, const float3 wo)
    {
        ExtraBSDFProperties result;
        result.bsdfCount = 0;
        return result;
    }
};

struct ReplayMaterialInstance : IMaterialInstance
{
    IMaterialInstance baseMaterial;
    uint readIdx;

    __init(IMaterialInstance mi, uint idx)
    {
        baseMaterial = mi;
        readIdx = idx;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        if (readIdx == kUnmappedIndex)
            return float3(0.0f);

        if (gHavePythonEval)
        {
            return gPythonInference.samples[readIdx].getWeight();
        }
        else
        {
            return baseMaterial.eval(sd, wo, sg);
        }
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        if (readIdx == kUnmappedIndex)
            return false;

        MaterialSample ms = gPythonSample.samples[readIdx];
        float3 woLocal = ms.getWo();
        result.wo = sd.frame.fromLocal(woLocal);

        if (gForceCosineSampling)
        {
            result.pdf = max(woLocal.z, 0.0f) * M_1_PI;
        }
        else if (gHavePythonSampling)
        {
            result.pdf = ms.getPdf();
        }
        else
        {
            result.pdf = baseMaterial.evalPdf(sd, result.wo, useImportanceSampling);
        }

        float3 weight;
        if (gHavePythonEval)
        {
            weight = ms.getWeight();
        }
        else
        {
            weight = baseMaterial.eval(sd, result.wo, sg);
        }

        result.weight = result.pdf > 0.0f ? weight / result.pdf : float3(0.0f);
        result.lobeType = uint(LobeType::SpecularReflection);
        return result.pdf > 0.0f;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        if (readIdx == kUnmappedIndex)
            return 0.0f;

        MaterialSample ms = gPythonInference.samples[readIdx];

        float3 woLocal = ms.getWo();

        if (gForceCosineSampling)
        {
            return max(woLocal.z, 0.0f) * M_1_PI;
        }
        else if (gHavePythonSampling)
        {
            return ms.getPdf();
        }
        else
        {
            return baseMaterial.evalPdf(sd, sd.frame.fromLocal(woLocal), useImportanceSampling);
        }
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};
        p.roughness = 0.5f;
        return p;
    }

    uint getLobeTypes(const ShadingData sd) { return uint(LobeType::Specular); }

    ExtraBSDFProperties getExtraBSDFProperties(const ShadingData sd, const float3 wo)
    {
        ExtraBSDFProperties result;
        result.bsdfCount = 0;
        return result;
    }
};
