/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.Scene;
import Utils.Color.ColorHelpers;
import LocalEmissiveTriangle;

/** Collects emissive triangles from the scene representation and copies
    them to a compact local array for fast access during light sampling and evaluation.
*/
struct UpdateEmissiveTriangles
{
    // Resources.
    RWStructuredBuffer<PackedLocalEmissiveTriangle> emissiveTriangles;

    uint emissiveTriangleCount;

    void process(uint triangleIndex)
    {
        if (triangleIndex >= emissiveTriangleCount)
            return;

        EmissiveTriangle inTriangle = gScene.lightCollection.getTriangle(triangleIndex);

        LocalEmissiveTriangle outTriangle = {};
        outTriangle.pos = inTriangle.posW[0];
        outTriangle.normal = inTriangle.normal;
        outTriangle.edge0 = inTriangle.posW[1] - inTriangle.posW[0];
        outTriangle.edge1 = inTriangle.posW[2] - inTriangle.posW[0];
        outTriangle.radiance = luminance(gScene.lightCollection.getAverageRadiance(triangleIndex));

        // DEMO21
        if (Scene::kUseLightProfile)
        {
            outTriangle.useLightProfile = gScene.materials.getMaterialHeader(inTriangle.materialID).isLightProfileEnabled();
        }

        emissiveTriangles[triangleIndex] = outTriangle.pack();
    }
};

cbuffer CB
{
    UpdateEmissiveTriangles gUpdateEmissiveTriangles;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gUpdateEmissiveTriangles.process(dispatchThreadId.x);
}
