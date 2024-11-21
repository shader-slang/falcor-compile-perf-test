/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Raster program that integrates each textured emissive triangle.
    The host is expected to set _VIEWPORT_DIMENSION.

    The emission is integrated over the triangles by rasterizing them
    in texture space with one pixel per texel. Conservative rasterization
    is used and the coverage of edge texels is computed analytically.

    The integrated result is exact under the assumption that the emissive
    texture will be sampled using nearest filtering at the coarsest mip.
    If that's not the case texels further away may affect the result.

    Another assumption is that the texture transform is affine so that
    the triangle is not degenerate in texture space. Non-affine transforms
    can happen if two or three vertices have identical texture coordinates,
    or if all three vertices lie on a line in texture space.
    For such triangles, we approximate the average radiance as the average
    of the emission sampled at the three vertices.
*/
import Scene.Scene;
import Utils.Geometry.GeometryHelpers;

// Check that defines are set.
#ifndef _VIEWPORT_DIM
#error _VIEWPORT_DIM is not defined
#endif

ParameterBlock<LightCollection> gLightCollection;

RWByteAddressBuffer gTexelMax;          ///< Max over texels in fp32 format. Using raw buffer for fp32 atomics compatibility.
RWByteAddressBuffer gTexelSum;          ///< Sum over texels (RGB) + number of texels (A) in 64-bit fixed-point format. Using raw buffer for 64-bit atomics compatibility.
SamplerState        gPointSampler;      ///< Sampler for fetching individual texels with nearest filtering.

struct GsOut
{
    float2 uv                               : TexCoord;
    float2 texelPos                         : TexelPosition;
    float4 posH                             : SV_Position;

    nointerpolation uint triIdx             : TriangleIndex;
    nointerpolation uint materialID         : MaterialID;
    nointerpolation float2 vtxTexelPos[3]   : VertexTexelPositions;
};

/** Null vertex shader. We'll do all the work in the geometry shader.
*/
void vsMain()
{
}

/** Geometry shader.
    We place textured emissive triangles in texture space scaled so that we get one
    pixel shader execution per texel. The vertex positions are passed on to the pixel shader.
    Non-textured emissives are culled.
*/
[maxvertexcount(3)]
void gsMain(uint triIdx : SV_PrimitiveID, inout TriangleStream<GsOut> outStream)
{
    // Fetch emissive triangle.
    const EmissiveTriangle tri = gLightCollection.getTriangle(triIdx);

    // Check if triangle is textured. Cull non-textured triangles.
    if (!gScene.materials.isEmissiveTextured(tri.materialID)) return;

    // Query texture dimensions.
    BasicMaterialData md = gScene.materials.getBasicMaterialData(tri.materialID);
    TextureInfo texInfo = gScene.materials.getTextureInfo(md.texEmissive);
    if (min(texInfo.width, texInfo.height) == 0) return;

    // Get triangle's bounding box in texture coordinates.
    const float2 uvMin = min(min(tri.texCoords[0], tri.texCoords[1]), tri.texCoords[2]);
    const float2 uvMax = max(max(tri.texCoords[0], tri.texCoords[1]), tri.texCoords[2]);
    const float2 uvOffset = floor(uvMin);

    GsOut gsOut;
    gsOut.triIdx = triIdx;
    gsOut.materialID = tri.materialID;

    for (int i = 0; i < 3; i++)
    {
        // Compute raster position in texture space.
        // We do this by offsetting UV so that it's always positive, scaled by texture dimension.
        float2 uv = tri.texCoords[i];
        gsOut.vtxTexelPos[i] = (uv - uvOffset) * float2(texInfo.width, texInfo.height); // Vertex pos in texels

        // TODO: Check if triangle is too large for viewport.
        // We need to detect that and handle it with a fallback mechanism (print error for now).
    }

    // Output vertices.
    for (int i = 0; i < 3; i++)
    {
        gsOut.uv = tri.texCoords[i];
        gsOut.texelPos = gsOut.vtxTexelPos[i];

        // Convert to 2DH normalized device coordinates.
        float2 ndcPos = gsOut.texelPos * (2.f / _VIEWPORT_DIM) - 1.f;
        gsOut.posH = float4(ndcPos, 0.f, 1.f); // Place triangle at w = 1.

        outStream.Append(gsOut);
    }
    outStream.RestartStrip();
}

/** Pixel shader.
    This is executed once per texel that the emissive triangle covers.
    We use conservative rasterization so texel centers do not necessarily need to be covered.
*/
void psMain(GsOut gsIn, uint innerCoverage : SV_InnerCoverage)
{
    // Fetch texel using nearest-neighbor sampling.
    BasicMaterialData md = gScene.materials.getBasicMaterialData(gsIn.materialID);
    float3 color = gScene.materials.sampleTexture(md.texEmissive, gPointSampler, gsIn.uv, 0.f).rgb; // Sample at mip 0

#if INTEGRATOR_PASS == 2
    // Compute texel coverage.
    // For partially covered texels, clip the triangle to the texel and compute the area analytically.
    float weight = 1.f;
    if (innerCoverage == 0)
    {
        float2 texelCorner = floor(gsIn.texelPos);
        float area = computeClippedTriangleArea2D(gsIn.vtxTexelPos, texelCorner, texelCorner + float2(1.f));
        weight = min(abs(area), 1.f); // The area may be negative due to winding.
    }

    // Weight the texel color by coverage.
    color *= weight;
#endif

    // We could use wave operations to reduce values across the warp before issuing global atomics.
    // However it would have to be done carefully since we have no guarantee that the GPU doesn't shade quads
    // from different triangles in the same warp. Hence we would have to perform reductions grouped by triIdx.
    // This code is currently not performance critical, so this optimization is left as a future exercise.

#if INTEGRATOR_PASS == 1
    // Determine maximum texel value.
    float maxVal = max(max(color.r, color.g), color.b);

    if (maxVal > 0.f)
    {
        uint address = gsIn.triIdx * 4;
        gTexelMax.InterlockedMax(address, asuint(maxVal));
    }

#elif INTEGRATOR_PASS == 2
    // Rescale texel values to unit range.
    float maxVal = asfloat(gTexelMax.Load(gsIn.triIdx * 4));
    float3 normalizedColor = maxVal > 0.f ? color / maxVal : float3(0.f);
    float4 val = float4(normalizedColor, weight);

    if (any(val > 0.f))
    {
        // Convert to 29.35 bit fixed point.
        // 29 bits is sufficient to represent the sum of 1.0 over the entire viewport (16K x 16K).
        const float scale = (1ull << 35);
        uint64_t4 f = uint64_t4(val * scale);

        // Accumulate using 64-bit atomics.
        uint address = gsIn.triIdx * 32;
        uint64_t originalValue;
        gTexelSum.InterlockedAdd64(address + 0, f.x, originalValue);
        gTexelSum.InterlockedAdd64(address + 8, f.y, originalValue);
        gTexelSum.InterlockedAdd64(address + 16, f.z, originalValue);
        gTexelSum.InterlockedAdd64(address + 24, f.w, originalValue);
    }
#else
    #error "Invalid value for INTEGRATOR_PASS"
#endif
}
