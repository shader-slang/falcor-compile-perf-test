/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import ScreenSpaceReSTIR.ScreenSpaceReSTIR;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDHelpers;
import LoadShadingData;
import InternalPathTracer;
import NRDHelpers;

/** Extends the InternalPathTracer with variants of handleHit() to trace NRD guide paths.
*/
extension InternalPathTracer
{
    /** Handle hit on delta reflection materials.
        After handling the hit, the path is terminated.
        Executed only for guide paths.
        \param[in,out] path The path state.
    */
    void handleDeltaReflectionHit(inout PathState path)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Write out reflectance/normalWRough/posW of the second path vertex
        // - Terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, lod);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path))
        {
            float3 originOffset = sd.posW - path.origin;

            float4x4 originOffsetMat = float4x4(
                1.f, 0.f, 0.f, originOffset.x,
                0.f, 1.f, 0.f, originOffset.y,
                0.f, 0.f, 1.f, originOffset.z,
                0.f, 0.f, 0.f, 1.f
            );

            path.imageXform = mul(path.imageXform, originOffsetMat);

            return;
        }

        // Create material instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        // Query BSDF lobes.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;

        const MaterialType materialType = sd.mtl.getMaterialType();

        if (isPrimaryHit)
        {
            // Terminate without the write-out if the path doesn't start as delta reflection.
            bool hasDeltaReflectionLobe = ((lobeTypes & (uint)LobeType::DeltaReflection) != 0);
            if (!hasDeltaReflectionLobe)
            {
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength, 0.f);
                path.terminate();
                return;
            }

            // Add primary ray length to the path length.
            float primaryHitDist = length(sd.posW - path.origin);
            path.sceneLength += float16_t(primaryHitDist);
            // Hijack pdf that we don't need.
            path.pdf += primaryHitDist;

            // Set the active lobes only to delta reflection on the first bounce.
            sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection);
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * bsdfProperties.emission;

            // Terminate after scatter ray on last vertex has been processed or non-delta lobe exists.
            const bool lastVertex = hasFinishedSurfaceBounces(path);
            const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;
            const bool isEmissive = any(bsdfProperties.emission > 0.f);

            if (lastVertex || hasNonDeltaLobes || isEmissive)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                const HitInfo primaryHit = HitInfo(vbuffer[pixel]);
                float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, reflectance, emission, bsdfProperties.guideNormal, bsdfProperties.roughness, float(path.sceneLength), hitDist, imageMotionVector);

                path.terminate();
                return;
            }

            // For glass in reflections, force guide paths to always follow transmission/reflection based on albedos.
            // This is pretty hacky but works best our of the possible options.
            // Stable guide buffers are a necessity.
            if (bsdfProperties.isTransmissive && all(bsdfProperties.specularReflectionAlbedo <= bsdfProperties.specularTransmissionAlbedo))
            {
                sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission);
            }
            else
            {
                sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection);
            }
        }

        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI && path.getVertexIndex() == 1)
        {
            path.reSTIRGIData.isDisabled = true;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeRayOrigin();

        // Hijack pdf that we don't need.
        float primaryHitDist = path.pdf;

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        path.pdf = primaryHitDist;

        if (!valid)
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength, 0.f);
            path.terminate();
            return;
        }

        // Terminate if transmission lobe was chosen but volume absorption is too high
        // but store the previous vertex shading data.
        if (path.isTransmission())
        {
            // Fetch volume absorption from the material. This field only exist in basic materials for now.
            bool semiOpaque = false;
            if (gScene.materials.isBasicMaterial(sd.materialID))
            {
                BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
                // TODO: Expose this arbitrary value as a constant.
                semiOpaque = any(md.volumeAbsorption > 100.f);
            }

            if (semiOpaque)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                const HitInfo primaryHit = HitInfo(vbuffer[pixel]);
                float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, reflectance, emission, bsdfProperties.guideNormal, bsdfProperties.roughness, float(path.sceneLength), hitDist, imageMotionVector);

                path.terminate();
                return;
            }
        }

        float3 originOffset = sd.posW - path.origin;
        // Update image xform
        float4x4 T;

        if (path.isTransmission())
        {
            float insideIoR = gScene.materials.evalIoR(sd.materialID);
            float rcpEta = sd.frontFacing ? (insideIoR / sd.IoR) : (sd.IoR / insideIoR);
            float normalCurvature = kDeltaTransmissionCurvatureMvecs ? sd.normalCurvature : 0.f;
            // For refractive (double sided) materials bsdfProperties.guideNormal point always in the same direction as view vector.
            T = getRefractionXform(path.origin, viewDir, bsdfProperties.guideNormal, rcpEta, normalCurvature, originOffset);
        }
        else
        {
            float normalCurvature = kDeltaReflectionCurvatureMvecs ? sd.normalCurvature : 0.f;
            T = getReflectionXform(path.origin, viewDir, bsdfProperties.guideNormal, normalCurvature, originOffset);
        }

        path.imageXform = mul(path.imageXform, T);
    }

    /** Handle hit on delta transmission materials.
        After handling the hit, a new scatter (delta transmission only) ray is generated or the path is terminated.
        Executed only for guide paths.
        \param[in,out] path The path state.
    */
    void handleDeltaTransmissionHit(inout PathState path)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Write out albedo/normal/posW on the first hit of non delta transmission BSDF lobe
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, lod);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path))
        {
            float3 originOffset = sd.posW - path.origin;

            float4x4 originOffsetMat = float4x4(
                1.f, 0.f, 0.f, originOffset.x,
                0.f, 1.f, 0.f, originOffset.y,
                0.f, 0.f, 1.f, originOffset.z,
                0.f, 0.f, 0.f, 1.f
            );

            path.imageXform = mul(path.imageXform, originOffsetMat);

            return;
        }

        // Create material instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        const uint lobeTypes = mi.getLobeTypes(sd);

        // Terminate without the write-out if the path doesn't start as delta transmission.
        const bool hasDeltaTransmissionLobe = ((lobeTypes & (uint)LobeType::DeltaTransmission) != 0);
        if (isPrimaryHit && !hasDeltaTransmissionLobe)
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, 0.f, 0.f);
            path.terminate();
            return;
        }

        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI && path.getVertexIndex() == 1)
        {
            path.reSTIRGIData.isDisabled = true;
        }

        if (isPrimaryHit)
        {
            // Add primary ray length to the path length.
            path.sceneLength += float16_t(length(sd.posW - path.origin));
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * bsdfProperties.emission;
        }

        // Terminate the delta transmission path.
        const bool lastVertex = hasFinishedSurfaceBounces(path);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        // Fetch volume absorption from the material. This field only exist in basic materials for now.
        bool semiOpaque = false;
        if (gScene.materials.isBasicMaterial(sd.materialID))
        {
            BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
            // TODO: Expose this arbitrary value as a constant.
            semiOpaque = any(md.volumeAbsorption > 100.f);
        }

        const MaterialType materialType = sd.mtl.getMaterialType();
        const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;

        if (lastVertex || semiOpaque)
        {
            float3 emission = path.L;
            const HitInfo primaryHit = HitInfo(vbuffer[pixel]);
            float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties), emission, bsdfProperties.guideNormal, bsdfProperties.roughness, float(path.sceneLength), imageMotionVector, sd.posW);

            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeRayOrigin();

        // Set the active lobes only to delta transmission.
        sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission);

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        // Delta transmission was not possible, fallback to delta reflection if it's allowed.
        if (!valid && isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
        {
            sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission | (uint)LobeType::DeltaReflection);
            valid = generateScatterRay(sd, mi, path);
        }

        if (!valid)
        {
            float3 emission = path.L;
            const HitInfo primaryHit = HitInfo(vbuffer[pixel]);
            float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties), emission, bsdfProperties.guideNormal, bsdfProperties.roughness, float(path.sceneLength), imageMotionVector, sd.posW);

            path.terminate();
            return;
        }

        float3 originOffset = sd.posW - path.origin;
        // Update image xform
        float4x4 T;

        if (path.isTransmission())
        {
            float insideIoR = gScene.materials.evalIoR(sd.materialID);
            float rcpEta = sd.frontFacing ? (insideIoR / sd.IoR) : (sd.IoR / insideIoR);
            float normalCurvature = kDeltaTransmissionCurvatureMvecs ? sd.normalCurvature : 0.f;
            // For refractive (double sided) materials bsdfProperties.guideNormal point always in the same direction as view vector.
            T = getRefractionXform(path.origin, viewDir, bsdfProperties.guideNormal, rcpEta, normalCurvature, originOffset);
        }
        else
        {
            float normalCurvature = kDeltaReflectionCurvatureMvecs ? sd.normalCurvature : 0.f;
            T = getReflectionXform(path.origin, viewDir, bsdfProperties.guideNormal, normalCurvature, originOffset);
        }

        path.imageXform = mul(path.imageXform, T);
    }
}
