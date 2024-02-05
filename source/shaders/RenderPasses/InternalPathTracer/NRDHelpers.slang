/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.IsotropicGGX;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDHelpers;
import Scene.Scene;
import Scene.ShadingData;
import Utils.Math.MathHelpers;
import RenderPasses.InternalPathTracer.PathState;

 /** Generate an orthonormal basis that tries to align tangent with the view vector.
     \param[in] v Unit view vector.
     \param[in] n Unit normal vector.
     \param[out] t Unit tangent vector.
     \param[out] b Unit bitangent vector.
 */
void buildOrthonormalBasis(float3 v, float3 n, out float3 t, out float3 b)
{
    static const float kEpsilon = 1.e-6f;

    t = abs(1.f + dot(n, v)) < kEpsilon ? float3(n.z, -n.x, n.y) : v;
    t -= n * dot(n, t);
    t = normalize(t);
    b = cross(n, t);
}

/** Wrap the xform so it operates in local coordinate system.
    \param[in] p World position of a point on a plane.
    \param[in] v Unit view vector.
    \param[in] n Unit normal vector.
    \param[in] L Xform.
    \return T Wrapped xform.
*/
float4x4 wrapInLocalXform(float3 p, float3 v, float3 n, float4x4 L)
{
    float3 t1, t2;
    buildOrthonormalBasis(v, n, t1, t2);

    // TODO: Optimize these matrix multiplies.

    float3x3 R = float3x3(
        t1.x, t2.x, n.x,
        t1.y, t2.y, n.y,
        t1.z, t2.z, n.z
    );

    // LocalToWorld xform.
    float4x4 P = float4x4(
        float4(R[0], p.x),
        float4(R[1], p.y),
        float4(R[2], p.z),
        float4(0.f, 0.f, 0.f, 1.f),
        );

    float3x3 invR = transpose(R);
    float3 t = -mul(invR, p);
    // WorldToLocal xform.
    float4x4 invP = float4x4(
        float4(invR[0], t.x),
        float4(invR[1], t.y),
        float4(invR[2], t.z),
        float4(0.f, 0.f, 0.f, 1.f),
        );

    float4x4 T = mul(L, invP);
    T = mul(P, T);
    return T;
}

/** Generate reflection xform that takes normal curvature into account.
    \param[in] p World position of a point on a plane.
    \param[in] v Unit view vector.
    \param[in] n Unit normal vector.
    \param[in] normalCurvature Normal curvature.
    \return T Reflection xform.
*/
float4x4 getReflectionXform(float3 p, float3 v, float3 n, float normalCurvature, float3 originOffset)
{
    // Planar reflection xform.
    float4x4 L = float4x4(
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, -1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    );

    if (abs(normalCurvature) > 1.e-6f)
    {
        L = float4x4(
            -1.f, 0.f, 0.f, 0.f,
            0.f, -1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, -2.f * normalCurvature, -1.f
        );
    }

    float4x4 originOffsetMat = float4x4(
        1.f, 0.f, 0.f, originOffset.x,
        0.f, 1.f, 0.f, originOffset.y,
        0.f, 0.f, 1.f, originOffset.z,
        0.f, 0.f, 0.f, 1.f
    );

    return mul(originOffsetMat, wrapInLocalXform(p, v, n, L));
}

/** Generate refraction xform that takes normal curvature into account.
    \param[in] p World position of a point on a plane.
    \param[in] v Unit view vector.
    \param[in] n Unit normal vector (assuming that dot(v, n) >= 0).
    \param[in] ior Relative index of refraction (transmissive IoR / incident IoR).
    \param[in] normalCurvature Normal curvature.
    \return T Refraction xform.
*/
float4x4 getRefractionXform(float3 p, float3 v, float3 n, float ior, float normalCurvature, float3 originOffset)
{
    const float cos_i = dot(v, n);
    // sint = sini / ior
    // 1 - cost^2 = (1 - cosi^2) / ior^2
    // 1 - (1 - cosi^2) / ior^2 = cost^2
    const float tmp = mad(mad(cos_i, -cos_i, 1.f), -1.f / (ior * ior), 1.f);
    const float cos_t = sqrt(max(0.f, tmp));

    const float l1 = cos_t == 0.f ? 1.f : cos_i;
    const float l2 = cos_t == 0.f ? -1.f : (cos_t * ior);

    // Planar refraction xform.
    float4x4 L = float4x4(
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, l1 / l2, 0.f,
        0.f, 0.f, 0.f, 1.f
    );

    if (abs(normalCurvature) > 1.e-6f)
    {
        L = float4x4(
            l2, 0.f, 0.f, 0.f,
            0.f, l2, 0.f, 0.f,
            0.f, 0.f, l1, 0.f,
            0.f, 0.f, (l2 - l1) * normalCurvature, l2
        );
    }

    float4x4 originOffsetMat = float4x4(
        1.f, 0.f, 0.f, originOffset.x,
        0.f, 1.f, 0.f, originOffset.y,
        0.f, 0.f, 1.f, originOffset.z,
        0.f, 0.f, 0.f, 1.f
    );

    return mul(originOffsetMat, wrapInLocalXform(p, v, n, L));
}

float2 calcImageMotionVector(const PathState path, const uint2 frameDim, const HitInfo primaryHit)
{
    float3 prevPosW = gScene.getPrevPosW(path.hit);
    float4 prevImagePosW = mul(path.imageXform, float4(prevPosW, 1.f));
    prevImagePosW /= prevImagePosW.w;

    // Compute motion vector in screen and world space.
    float2 pixelPos = path.getPixel() + float2(0.5f, 0.5f);
    float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevImagePosW.xyz, 1.f));
    float2 imageMotionVector = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector

    // Fall back to the primary surface motion vector.
    if (any(isnan(imageMotionVector) || isinf(imageMotionVector)))
    {
        float3 prevPrimaryPosW = gScene.getPrevPosW(primaryHit);
        float4 prevPrimaryPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPrimaryPosW.xyz, 1.f));
        imageMotionVector = calcMotionVector(pixelPos, prevPrimaryPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
    }

    return imageMotionVector;
}

void setNRDPrimaryHitEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const float3 emission)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        // Generate primary hit guide buffers.
        if (useNRDDemodulation)
        {
            outputNRD.primaryHitEmission[pixel] = float4(emission, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitEmission[pixel] = 0.f;
        }
    }
}

void setNRDPrimaryHitReflectance(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const ShadingData sd, const BSDFProperties bsdfProperties)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        // Generate primary hit guide buffers.
        if (useNRDDemodulation)
        {
            // Use sum of reflection/transmission albedo as they are denoised together.
            const float3 diffuseReflectance = max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            outputNRD.primaryHitDiffuseReflectance[pixel] = float4(diffuseReflectance, 1.f);

            // We use an approximate shading model with a single specular lobe.
            const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
            specularReflectance = max(kNRDMinReflectance, specularReflectance);
            outputNRD.primaryHitSpecularReflectance[pixel] = float4(specularReflectance, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitDiffuseReflectance[pixel] = 1.f;
            outputNRD.primaryHitSpecularReflectance[pixel] = 1.f;
        }
    }
}

void setNRDPrimaryHitSeparatedRadiance(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const bool isPrimaryHit, const uint outSampleIdx)
{
    if (isPrimaryHit)
    {
        if (useNRDDemodulation && path.isDeltaReflectionPrimaryHit())
        {
            outputNRD.samplePrimaryHitNEEOnDelta[outSampleIdx] = float4(path.L, 0);
        }
        else
        {
            outputNRD.samplePrimaryHitNEEOnDelta[outSampleIdx] = float4(0.f);
        }
    }
}

void setNRDSampleHitDist(NRDBuffers outputNRD, const PathState path, const uint outSampleIdx)
{
    if (path.getVertexIndex() == 2)
    {
        outputNRD.sampleHitDist[outSampleIdx] = float(path.sceneLength);
    }
}

void setNRDSampleEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint outSampleIdx, const bool isPrimaryHit, const float3 emission, const bool wasDeltaOnlyPathBeforeScattering)
{
    if (useNRDDemodulation)
    {
        // Always demodulate emission on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            outputNRD.sampleEmission[outSampleIdx] = float4(emission, 1.f);
        }
        // Additionally, accumulate emission along the delta path.
        else if ((path.isDeltaReflectionPrimaryHit() || path.isDeltaTransmissionPath()) && any(emission > 0.f))
        {
            const bool demodulateDeltaReflectionEmission = path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering;
            const bool demodulateDeltaTransmissionEmission = path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering;
            if (demodulateDeltaReflectionEmission || demodulateDeltaTransmissionEmission)
            {
                float3 prevEmission = outputNRD.sampleEmission[outSampleIdx].rgb;
                outputNRD.sampleEmission[outSampleIdx] = float4(prevEmission + emission, 1.f);
            }
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.sampleEmission[outSampleIdx] = 0.f;
    }
}

void setNRDSampleReflectance(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint outSampleIdx, const bool isPrimaryHit, const ShadingData sd, const BSDFProperties bsdfProperties, const uint lobeTypes, const bool wasDeltaOnlyPathBeforeScattering)
{
    // Demodulate reflectance.
    if (useNRDDemodulation)
    {
        const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        // Always demodulate reflectance from diffuse and specular paths on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            if (path.isDiffusePrimaryHit())
            {
                // Use sum of reflection/transmission albedo as they are denoised together.
                const float3 diffuseReflectance = max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
                outputNRD.sampleReflectance[outSampleIdx] = float4(diffuseReflectance, 1.f);
            }
            else if (path.isSpecularPrimaryHit())
            {
                // We use an approximate shading model with a single specular lobe.
                const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
                const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
                float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
                specularReflectance = max(kNRDMinReflectance, specularReflectance);
                outputNRD.sampleReflectance[outSampleIdx] = float4(specularReflectance, 1.f);
            }
            else
            {
                outputNRD.sampleReflectance[outSampleIdx] = 1.f;
            }
        }
        // Demodulate reflectance from the second vertex along delta reflection paths.
        else if (path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            const MaterialType materialType = sd.mtl.getMaterialType();
            const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);

            outputNRD.sampleReflectance[outSampleIdx] = float4(deltaReflectance, 1.f);
        }
        // Demodulate reflectance from the first non-delta vertex along delta transmission paths.
        else if (path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            const MaterialType materialType = sd.mtl.getMaterialType();
            const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);

            outputNRD.sampleReflectance[outSampleIdx] = float4(deltaReflectance, 1.f);
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.sampleReflectance[outSampleIdx] = 1.f;
    }
}

/** Write out delta reflection guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaReflectionGuideBuffers(NRDBuffers outputNRD, const bool useNRDDemodulation, const uint2 pixelPos, float3 reflectance, float3 emission, float3 normal, float roughness, float pathLength, float hitDist, float2 motionVector)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    outputNRD.deltaReflectionEmission[pixelPos] = float4(emission, 0.f);
    outputNRD.deltaReflectionReflectance[pixelPos] = float4(reflectance, 0.f);
    outputNRD.deltaReflectionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    outputNRD.deltaReflectionPathLength[pixelPos] = pathLength;
    outputNRD.deltaReflectionHitDist[pixelPos] = hitDist;
    outputNRD.deltaReflectionMotionVectors[pixelPos] = motionVector;
}

/** Write out delta transmission guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaTransmissionGuideBuffers(NRDBuffers outputNRD, const bool useNRDDemodulation, const uint2 pixelPos, float3 reflectance, float3 emission, float3 normal, float roughness, float pathLength, float2 motionVector, float3 posW)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    outputNRD.deltaTransmissionEmission[pixelPos] = float4(emission, 0.f);
    outputNRD.deltaTransmissionReflectance[pixelPos] = float4(reflectance, 0.f);
    outputNRD.deltaTransmissionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    outputNRD.deltaTransmissionPathLength[pixelPos] = pathLength;
    outputNRD.deltaTransmissionMotionVectors[pixelPos] = motionVector;
    outputNRD.deltaTransmissionPosW[pixelPos] = float4(posW, 0.f);
}
