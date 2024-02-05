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
 * Helper functions for loading G/V-buffer data.
 *
 * loadShadingData() loads everything and prepares the ShadingData struct.
 *
 * It is assumed the host sets up a define for all optional input buffers:
 * is_valid_<name> is 1 if buffer with this name is bound, 0 otherwise.
 */

__exported import Scene.Shading;
import Scene.Material.ShadingUtils;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import RenderPasses.PythonPathTracer.TraceReplayMaterials;
import RenderPasses.PythonPathTracer.FilterHelpers;

#if USE_VBUFFER
// V-buffer inputs
Texture2D<PackedHitInfo> gVBuffer;
#else
// G-buffer inputs
Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldShadingTangent;
Texture2D<float4> gWorldFaceNormal;
Texture2D<float4> gWorldView; // Optional
Texture2D<float2> gTextureCoord;
Texture2D<float4> gTextureGrads; // Optional
Texture2D<uint4> gMaterialData;
Texture2D<PackedHitInfo> gVBuffer; // Optional
#endif

#define is_valid(name) (is_valid_##name != 0)

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 pixel, uint2 frameDim, const Camera camera)
{
#if !USE_VBUFFER
    if (is_valid(gWorldView))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gWorldView[pixel].xyz;
    }
    else
#endif
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(pixel, frameDim).dir;
    }
}

/**
 * Helper to create a texture sampler instance.
 * The method for computing texture level-of-detail depends on the configuration.
 * @param[in] pixel Current pixel coordinates.
 * @return Texture sampler instance.
 */
ITextureSampler createTextureSampler(const uint2 pixel)
{
#if USE_VBUFFER
    return ExplicitLodTextureSampler(0.f);
#else
    if (is_valid(gTextureGrads))
    {
        float4 texGrads = gTextureGrads[pixel];
        return ExplicitGradientTextureSampler(texGrads.xy, texGrads.zw);
    }
    else
    {
        return ExplicitLodTextureSampler(0.f);
    }
#endif
}

/**
 * Helper for setting up the ShadingData struct based on loaded data.
 * @param[in] pixel Current pixel coordinates.
 * @param[in] frameDim Frame dimensions in pixel.
 * @param[in] camera Current camera.
 * @param[in] lod Method for computing texture level-of-detail.
 * @param[out] sd ShadingData struct.
 * @param[out] hit HitInfo struct returned with geometry fetched from vbuffer if available. Only valid if true is returned.
 * @return True if the pixel has valid data (not a background pixel). Note sd.V is always valid.
 */
bool loadShadingData(
    const uint2 pixel,
    const uint2 frameDim,
    const Camera camera,
    const ITextureSampler lod,
    out ShadingData sd,
    out HitInfo hit,
    inout SampleGenerator sg
)
{
    sd = {};
    hit = {};

    float3 rayDir = getPrimaryRayDir(pixel, frameDim, camera);
    bool valid = false;

#if USE_VBUFFER
    hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        // Evaluate Falcor's material parameters at the hit point.
        // TODO: Implement texLOD to enable texture filtering in prepareShadingData().
        StaticVertexData vertices[3];
        VertexData v = gScene.getVertexData(triangleHit, vertices);

        bool isPython = gIsPythonMaterial[materialID];
        if (!isPython)
        {
            float2 ellipseX, ellipseY;
            computeFilterEllipse(v, vertices, rayDir, length(camera.getPosition() - v.posW), ellipseX, ellipseY);
            sampleFilterFootprint(ellipseX, ellipseY, v, sg);
        }
        else
        {
            // Consume random numbers without using result to equal out RNG usage
            sampleFilterFootprint(float2(0.0f), float2(0.0f), float2(0.0f), sg);
        }

        // Prepare shading data.
        bool useNormalMap = !isPython || gUseNormalMap; // TODO: Cleanup
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

        valid = true;
    }
#else
    float4 worldPos = gWorldPosition[pixel];
    if (worldPos.w != 0.f) // Using w to indicate valid geometry for now.
    {
        // Load G-buffer data.
        float3 normal = gWorldShadingNormal[pixel].xyz;
        float4 tangent = gWorldShadingTangent[pixel];
        float3 faceNormal = gWorldFaceNormal[pixel].xyz;
        float2 texC = gTextureCoord[pixel];
        uint materialID = gMaterialData[pixel].x;

        VertexData v = {};
        v.posW = worldPos.xyz;
        v.normalW = normalize(normal);
        v.tangentW = float4(normalize(tangent.xyz), tangent.w);
        v.faceNormalW = normalize(faceNormal);
        v.texC = texC;
        v.coneTexLODValue = 0.f;

        // Load hit info from V-buffer if available. This is needed for ray footprint.
        bool isPython = gIsPythonMaterial[materialID];
        if (is_valid(gVBuffer))
        {
            hit = HitInfo(gVBuffer[pixel]);

            if (!isPython)
            {
                StaticVertexData vertices[3];
                VertexData v = gScene.getVertexData(hit.getTriangleHit(), vertices);

                float2 ellipseX, ellipseY;
                computeFilterEllipse(v, vertices, rayDir, length(camera.getPosition() - v.posW), ellipseX, ellipseY);
                sampleFilterFootprint(ellipseX, ellipseY, v, sg);
            }
            else
            {
                // Consume random numbers without using result to equal out RNG usage
                sampleFilterFootprint(float2(0.0f), float2(0.0f), float2(0.0f), sg);
            }
        }

        // Prepare shading data.
        bool useNormalMap = !isPython || gUseNormalMap; // TODO: Cleanup
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

        valid = true;
    }
#endif

    sd.V = -rayDir;
#ifdef FALCOR_INTERNAL
    sd.setHitInfo(hit);
#endif
    return valid;
}

/**
 * Helper for setting up the ShadingData struct based on loaded data.
 * @param[in] pixel Current pixel coordinates.
 * @param[in] frameDim Frame dimensions in pixel.
 * @param[in] camera Current camera.
 * @param[in] lod Method for computing texture level-of-detail.
 * @param[out] sd ShadingData struct.
 * @return True if the pixel has valid data (not a background pixel). Note sd.V is always valid.
 */
bool loadShadingData(
    const uint2 pixel,
    const uint2 frameDim,
    const Camera camera,
    const ITextureSampler lod,
    out ShadingData sd,
    inout SampleGenerator sg
)
{
    HitInfo hit;
    return loadShadingData(pixel, frameDim, camera, lod, sd, hit, sg);
}
