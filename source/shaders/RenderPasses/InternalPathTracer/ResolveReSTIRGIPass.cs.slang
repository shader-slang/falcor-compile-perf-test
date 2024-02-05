/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Compute shader using ReSTIR GI output sample to calculate indirect illumination.
*/
import Utils.Color.ColorHelpers;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDData;
import RenderPasses.Shared.Denoising.NRDHelpers;
import ScreenSpaceReSTIR.ScreenSpaceReSTIR;
import LoadShadingData;
import Params;
import Utils.Sampling.TinyUniformSampleGenerator;

struct ResolveReSTIRGIPass
{
    PathTracerParams params;
    ScreenSpaceReSTIR screenSpaceReSTIR;
    Texture2D<uint> sampleOffset;                           ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.
    RWTexture2D<float4> outputColor;                        ///< Output color buffer if kSamplesPerPixel == 1.
    Texture2D<PackedHitInfo> vbuffer;                       ///< Fullscreen V-buffer for the primary hits.
    RWStructuredBuffer<NRDRadiance> sampleNRDRadiance;
    RWStructuredBuffer<float> sampleNRDHitDist;
    StructuredBuffer<float4> sampleNRDReflectance;

    /** Entry point for resolve ReSTIR GI pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        HitInfo hit;
        hit = HitInfo(vbuffer[pixel]);
        if (!hit.isValid()) return;

        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, screenSpaceReSTIR.frameDim);
        let lod = ExplicitLodTextureSampler(0.f);
        ShadingData sd = loadShadingData(hit, cameraRay.origin, cameraRay.dir, lod);

        // Create material instance and query its properties.
        let hints = getMaterialInstanceHints(hit, true /* primary hit */);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        // ReSTIR uses a simple material model with only diffuse and specular reflection lobes.
        // We query the BSDF for the diffuse albedo and specular reflectance, and use their luminances as weights.
        // TODO: https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1406
        float roughness = max(0.08f, bsdfProperties.roughness); // TODO: Remove/move clamp into ReSTIR.
        float ggxAlpha = roughness * roughness;
        if (bsdfProperties.isTransmissive ||
            ScreenSpaceReSTIR::ignoreReSTIRGI(ggxAlpha, luminance(bsdfProperties.diffuseReflectionAlbedo), luminance(bsdfProperties.specularReflectance)))
        {
            return;
        }

        float3 color = float3(0.f, 0.f, 0.f);
        float hitT = 0.f;
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, params.frameCount);
        for (int i = 0; i < screenSpaceReSTIR.giReservoirCount; i++)
        {
            float3 creationPoint;
            float3 creationNormal;
            float3 position;
            float3 normal;
            float3 radiance;
            float avgWeight;
            screenSpaceReSTIR.getGIFinalSample(pixel, i, creationPoint, creationNormal, position, normal, radiance, avgWeight);

            float3 L = normalize(position - sd.posW);
            float3 diffuseBSDF = mi.eval(sd, L, sg);

            float W = avgWeight;
            float3 diffuseRadiance = radiance * diffuseBSDF * W;
            color += diffuseRadiance;
            hitT += length(position - sd.posW);
        }
        color /= float(screenSpaceReSTIR.giReservoirCount);
        hitT /= float(screenSpaceReSTIR.giReservoirCount);

        const uint outIdx = params.getSampleOffset(pixel, sampleOffset);
        NRDRadiance nrdRadiance = sampleNRDRadiance[outIdx];
        if (nrdRadiance.getPathType() == NRDPathType::Diffuse)
        {
            float3 reflectance = sampleNRDReflectance[outIdx].rgb;
            nrdRadiance.radiance += color / reflectance;
            sampleNRDRadiance[outIdx] = nrdRadiance;
            sampleNRDHitDist[outIdx] = hitT;
        }

        if (kSamplesPerPixel == 1)
        {
            // Accumulate color directly to frame buffer.
            outputColor[pixel].rgb += color.rgb;
        }
    }
};

cbuffer CB
{
    ResolveReSTIRGIPass gResolveReSTIRGIPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gResolveReSTIRGIPass.execute(dispatchThreadId.xy);
}
