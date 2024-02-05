/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Scene.Shading;
import Scene.Material.ShadingUtils;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import RenderPasses.PythonPathTracer.FilterHelpers;

#include "Utils/Math/MathConstants.slangh"
#include "MaterialDataShared.slang"

/** Material data generator.

    The scene materials are sampled to generate material data.
    The generated data is written to a buffer for readback to the host.

    The kernel is dispatched over a 3D grid of threads.
    The threads are currently arranged as a grid in (u, v) where each texel
    samples incident directions over a grid of (theta, phi) directions.
    The third dimension is the sample index within the grid cell.
    The dispatch dimensions are thus (u * theta, v * phi, sampleCount).
*/
struct MaterialDataGenerator
{
    uint4 gridDim;                 ///< Sampling grid dimensions in (u, v, thetaI, phiI).
    uint sampleCount;              ///< Number of samples per grid cell.
    float4 uvRange;                ///< Range in uv-space to sample (uMin, vMin, uMax, vMax).
    uint useNormalMapping;         ///< Whether normal mapping should be included as part of the material evaluation.
    uint materialID;               ///< Which material to sample.
    uint batchIndex;               ///< Index of current sample batch.
    uint sampleTexelCenter;        ///< Whether to disable spatial jittering within the grid cells.
    uint sampleBufferContent;      ///< Flag describing valid data (not to be sampled) in the sampleBuffer.
    float permuteChannelsRatio;    ///< Ratio of random color channels permutation for the evaluated materials.
    float balancedSamplingRatio;   ///< Sample ratio used for the "balanced" material sampling strategies.
    uint materialSamplingStrategy; ///< Which of the material sampling strategies should be used.

    uint sourceResolution;          ///< Largest material texture resolution. // TODO: Can we get this automatically?
    uint2 minMaxMipLevel;           ///< Minimum and maximum MIP level that are sampled for LOD training.
    float angularPerturbationWidth; ///< Half angle of the perturbation cone (for mollification).
    uint2 numPerturbationSamples;   ///< Minimum and maximum number of perturbation samples.

    uint skipEvaluation; ///< Skip actual material evaluation, but still fetch material attributes.

    bool sampleContains(SampleBufferContentDesc flag, SampleBufferContentDesc sampleContent)
    {
        if (flag == SampleBufferContentDesc::None)
            return sampleContent == SampleBufferContentDesc::None;
        else
            return (uint(sampleContent) & uint(flag)) == uint(flag);
    }

    MaterialSample generateMaterialSample(
        uint3 threadId,
        uint2 threadGroupSize,
        SampleBufferContentDesc fixedInputs,
        float2 uvFixed,
        float3 wiFixed,
        float3 woFixed
    )
    {
        // Map thread ID to texel grid cell indices.
        uint2 uvIdx = threadId.xy / gridDim.zw;
        uint sampleIdx = threadId.z;

        // Check if the sample buffer contains uvs or directions that we should use (instead of sampling them).
        float2 uv = float2(0.0f);
        float3 wi = float3(0.0f);
        float3 wo = float3(0.0f);
        if (!sampleContains(SampleBufferContentDesc::None, fixedInputs))
        {
            if (sampleContains(SampleBufferContentDesc::UV, fixedInputs))
                uv = uvFixed;
            if (sampleContains(SampleBufferContentDesc::Wi, fixedInputs))
                wi = wiFixed;
            if (sampleContains(SampleBufferContentDesc::Wo, fixedInputs))
                wo = woFixed;
        }

        // Create pseudorandom number generator seeded by thread ID and sample batch index.
        uint seed = batchIndex * sampleCount + sampleIdx;
        SampleGenerator sg = SampleGenerator(threadId.xy, seed);

        // Sample uvs if necessary.
        if (!sampleContains(SampleBufferContentDesc::UV, fixedInputs))
            uv = sampleUVs(uvIdx, sg);

        // Sample directions if necessary.
        bool success = true;
        bool hasWo = sampleContains(SampleBufferContentDesc::Wo, fixedInputs);
        if (!sampleContains(SampleBufferContentDesc::Wi, fixedInputs))
            success = sampleDirections(sg, wi, wo, hasWo);

        // Create material instance (requires wi)
        let lod = ExplicitLodTextureSampler(0.f);
        ShadingData sd = setupShadingData(uv, wi, lod);

        // Choose camera position
        float mipLevelShift = 1.0f;
        SampleGenerator mipGen = SampleGenerator((threadId.xy / threadGroupSize) * threadGroupSize, seed);
        float mipXi = sampleNext1D(mipGen);
        float mipLevel = minMaxMipLevel.x + -log2(1.0f - mipXi * (1.0f - exp2(-(minMaxMipLevel.y - minMaxMipLevel.x + 1.0f))));
        mipLevel = clamp(floor(mipLevel), minMaxMipLevel.x, minMaxMipLevel.y);
        float radius = mipLevel == 0.0f ? 0.0f : 0.5f * exp2(mipLevel + mipLevelShift) / sourceResolution;
        float camDist = 2.0f * radius / gScreenSpacePixelSpreadAngle;

        float3 cameraPos = sd.posW + sd.V * camDist;

        // Setup fake geometry information
        MaterialSampleGeometryInfo info = buildGeometryInfo(uv, cameraPos);
        info.lod = mipLevel;
        MaterialSample ms = MaterialSample(sd, info, wi, true);

        // Material samples may choose to ignore filter footprint. Get the sample's opinion and act accordingly
        ms.setFilterFootprint(float2(radius, 0.0f), float2(0.0f, radius), mipLevel);
        float2 ellipseX, ellipseY;
        ms.getFilterFootprint(ellipseX, ellipseY);
        bool deltaFootprint = (length(ellipseX) == 0.0f) && (length(ellipseY) == 0.0f);

        // Create material instance.
        const uint hints = !useNormalMapping ? (uint)MaterialInstanceHints::DisableNormalMapping : 0;
        IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);

        // Sample wo, if it hasn't been sampled in sampleDirections.
        if (hasWo)
        {
            ms.setWo(wo);
        }
        else
        {
            if (deltaFootprint)
            {
                success &= sampleWo(sd, mi, sg, ms);
            }
            else
            {
                float2 uv = sampleFilterFootprint(ellipseX, ellipseY, sd.uv, sg);
                ShadingData perturbed = setupShadingData(uv, wi, lod);
                IMaterialInstance mi = gScene.materials.getMaterialInstance(perturbed, lod, hints);
                if (!sampleWo(sd, mi, sg, ms))
                    ms.setWo(sample_hemisphere(sampleNext2D(sg)));
            }
        }

        // Evaluate the material
        if (success && !skipEvaluation)
        {
            ms.markValid();

            if (numPerturbationSamples[1] > 0 /*&& !deltaFootprint*/)
            {
                float2 uvBase = sd.uv;
                float3 woBase = ms.getWo();

                int maxPixelsInFootprint = max(int(sqr(radius * sourceResolution) * M_PI + 1.0f) / 2, 1);
                int numSamples = max(numPerturbationSamples[0], min(numPerturbationSamples[1], maxPixelsInFootprint));

                float3 sum = float3(0.0f);
                for (uint i = 0; i < numSamples; ++i)
                {
                    float2 uv = sampleFilterFootprint(ellipseX, ellipseY, uvBase, mipGen);
                    float3 wo = perturbDirection(woBase, cos(angularPerturbationWidth), sg);

                    ShadingData perturbed = setupShadingData(uv, wi, lod);
                    IMaterialInstance mi = gScene.materials.getMaterialInstance(perturbed, lod, hints);
                    sum += mi.eval(perturbed, wo, sg);
                }
                ms.setWeight(sum / numSamples);
            }
            else
            {
                ms.setWeight(mi.eval(sd, ms.getWo(), sg));
                ms.setPdf(mi.evalPdf(sd, ms.getWo(), true));
            }
        }

        // Always also evaluate the additional material properties.
        ms.setMaterialProperties(sd, mi, info.lod);

        // Permute color channels
        if (sampleNext1D(sg) < permuteChannelsRatio)
            ms.permuteChannels(sampleNext1D(sg));

        return ms;
    }

    float3 perturbDirection(float3 w, float cosTheta, inout SampleGenerator sg)
    {
        float3 wLocal = sample_cone(sampleNext2D(sg), cosTheta);

        float3 X = perp_stark(w);
        float3 Y = cross(w, X);
        return wLocal.x * X + wLocal.y * Y + wLocal.z * w;
    }

    float2 sampleUVs(const uint2 uvIdx, inout SampleGenerator sg)
    {
        float2 offset = sampleTexelCenter ? float2(0.5f) : sampleNext2D(sg);
        float2 s = (float2(uvIdx) + offset) / gridDim.xy; // s in [0,1)^2.
        return lerp(uvRange.xy, uvRange.zw, s);
    }

    bool sampleDirections(inout SampleGenerator sg, inout float3 wi, inout float3 wo, inout bool hasWo)
    {
        const NeuralMaterialSamplingStrategy strategy = NeuralMaterialSamplingStrategy(materialSamplingStrategy);

        if (strategy == NeuralMaterialSamplingStrategy::UniformRusinkiewicz ||
            (strategy == NeuralMaterialSamplingStrategy::BalancedRusinkiewicz && sampleNext1D(sg) < balancedSamplingRatio))
        {
            hasWo = true;
            return sampleUniformRusinkiewicz(sg, wi, wo);
        }
        else if (strategy == NeuralMaterialSamplingStrategy::UniformWiUniformWo)
        {
            wi = sample_hemisphere(sampleNext2D(sg));
            wo = sample_hemisphere(sampleNext2D(sg));
            hasWo = true;
            return true;
        }
        else
        {
            float2 s = sampleNext2D(sg);
            if (strategy == NeuralMaterialSamplingStrategy::CosineWiImportanceWo)
            {
                float pdf;
                wi = sample_cosine_hemisphere_concentric(s, pdf);
            }
            else
            {
                wi = sample_hemisphere(s);
            }
            return true;
        }
    }

    void rusinkiewicz_to_cartesian(float2 h_sph, float2 d_sph, out float3 wi, out float3 wo)
    {
        float3 d = spherical_to_cartesian_rad(d_sph);
        float3 temp = rotate_y(d, h_sph.x);
        wi = rotate_z(temp, h_sph.y);
        float3 half = spherical_to_cartesian_rad(h_sph);
        wo = normalize(2 * dot(wi, half) * half - wi);
    }

    bool sampleUniformRusinkiewicz(inout SampleGenerator sg, out float3 wi, out float3 wo)
    {
        bool success = false;
        uint tries = 0;

        while (!success && tries < 100)
        {
            float theta_h_warp_power = 1.f;
            float theta_h = pow(sampleNext1D(sg), theta_h_warp_power) * M_PI * 0.5f;
            float theta_d = sampleNext1D(sg) * M_PI * 0.5f;
            float phi_h = sampleNext1D(sg) * M_PI * 2.f;
            float phi_d = sampleNext1D(sg) * M_PI * 2.f;
            rusinkiewicz_to_cartesian(float2(theta_h, phi_h), float2(theta_d, phi_d), wi, wo);
            success = wi.z >= 0.f && wo.z >= 0.f;
            tries++;
        }

        return success;
    }

    MaterialSampleGeometryInfo buildGeometryInfo(float2 uv, float3 cameraPos)
    {
        // Setup fake vertex data describing the shading point and hit geometry.
        // We assume a flat quad made of two triangles (assuming UV is in [0, 1]^2)
        bool flip = uv.x + uv.y > 1.0f;
        float3 v0 = flip ? float3(1.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 0.0f);
        float3 v1 = flip ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
        float3 v2 = flip ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        StaticVertexData vertices[] = {
            { v0, float3(0.0f, 0.0f, 1.0f), float4(1.0f, 0.0f, 0.0f, 1.0f), v0.xy, 0.0f },
            { v1, float3(0.0f, 0.0f, 1.0f), float4(1.0f, 0.0f, 0.0f, 1.0f), v1.xy, 0.0f },
            { v2, float3(0.0f, 0.0f, 1.0f), float4(1.0f, 0.0f, 0.0f, 1.0f), v2.xy, 0.0f }
        };

        MaterialSampleGeometryInfo info;
        info.cameraPos = cameraPos;
        info.vertices = vertices;
        info.v = buildVertexData(uv);

        return info;
    }

    VertexData buildVertexData(float2 uv)
    {
        // Setup vertex data describing the shading.
        // Normally during rendering this would come from a ray hit in the scene.
        VertexData v = {};
        v.posW = float3(uv, 0);
        v.normalW = float3(0, 0, 1);
        v.tangentW = float4(1, 0, 0, 1);
        v.texC = uv;
        v.faceNormalW = v.normalW;
        v.curveRadius = 0.f;
        v.coneTexLODValue = 0.f;

        return v;
    }

    ShadingData setupShadingData(const float2 uv, const float3 wi, const ITextureSampler lod)
    {
        // Query the material system to prepare shading data.
        let v = buildVertexData(uv);
        return gScene.materials.prepareShadingData(v, materialID, wi, lod);
    }

    bool sampleWo(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, inout MaterialSample ms)
    {
        const NeuralMaterialSamplingStrategy strategy = NeuralMaterialSamplingStrategy(materialSamplingStrategy);

        if (strategy == NeuralMaterialSamplingStrategy::UniformWiBalancedWo)
        {
            // Generate a uniform direction.
            ms.setWo(sample_hemisphere(sampleNext1D(sg)));
            ms.setPdf(1.f / (2.f * M_PI));
            return true;
        }
        else
        {
            // Try importance sampling the material.
            // Note that the sample function may fail to generate a sample.
            // In that case we retry up to a maximum number of attempts.
            BSDFSample miSample = {};
            bool success = false;
            uint tries = 0;

            while (!success && tries < 1)
            {
                success = mi.sample(sd, sg, miSample);
                tries++;
            }
            ms.setWo(miSample.wo);
            ms.setPdf(miSample.pdf);
            return success;
        }
    }
};
