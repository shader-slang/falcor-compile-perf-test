/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import LoadShadingData;
import Utils.Color.ColorHelpers;
import ScreenSpaceReSTIR.ScreenSpaceReSTIR;

/** Prepares ReSTIR surface data.
 */
struct PrepareSurfaceData
{
    Texture2D<PackedHitInfo> vbuffer;

    ScreenSpaceReSTIR screenSpaceReSTIR;

    uint2 frameDim;

    void execute(uint2 pixel)
    {
        if (any(pixel >= frameDim))
            return;

        ShadingData sd;
        let lod = ExplicitLodTextureSampler(0.f); // TODO: Implement texture level-of-detail.
        bool isValidSurface = loadShadingData(pixel, frameDim, gScene.camera, vbuffer, lod, sd);

        if (isValidSurface)
        {
            // Create material instance and query its properties.
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            let bsdfProperties = mi.getProperties(sd);

            // ReSTIR uses a simple material model with only diffuse and specular reflection lobes.
            // We query the BSDF for the diffuse albedo and specular reflectance, and use their luminances as weights.
            // TODO: https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1406
            float depth = distance(gScene.camera.getPosition(), sd.posW);
            float diffuseWeight = luminance(bsdfProperties.diffuseReflectionAlbedo);
            float specularWeight = luminance(bsdfProperties.specularReflectance);
            screenSpaceReSTIR.setSurfaceData(
                pixel,
                sd.computeRayOrigin(),
                depth,
                bsdfProperties.guideNormal,
                sd.faceN,
                diffuseWeight,
                specularWeight,
                bsdfProperties.roughness
            );
        }
        else
        {
            screenSpaceReSTIR.setInvalidSurfaceData(pixel);
        }
    }
};

cbuffer CB
{
    PrepareSurfaceData gPrepareSurfaceData;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gPrepareSurfaceData.execute(dispatchThreadId.xy);
}
