/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.IMaterialInstance;
import Scene.Material.VolumeProperties;
import Rendering.Materials.Fresnel;
import Rendering.Volumes.PhaseFunction;
import Utils.Math.MathHelpers;
import Utils.Debug.PixelDebug;
import Utils.Color.ColorHelpers;

import MxWeights;
import Rendering.Materials.IBSDF;

/**
 * Generic class to handle all layering combinations.
 *
 * @tparam TBsdfCount Number of BSDFs, should always be greater than 0.
 * @tparam TLayerCount Number of layers, can be zero.
 * @tparam TWeightsCalc Interface to the generated code for computing layering weights, the layering program.
 */
struct MxLayeredMaterialInstance<
    let TBsdfCount : int,
    let TLayerCount : int,
    TLayeredMaterialData : IMxLayeredMaterialData,
    TWeightsCalc : IMxLayeredWeightCalculator> : IMaterialInstance
{
    TLayeredMaterialData data;
    TWeightsCalc calculator;
    typealias Weights = MxWeights<TBsdfCount>;

    bool hasVolumeProperties() { return data.hasSubsurfaceData(); }

    VolumeProperties getVolumeProperties()
    {
        /// Could also do something here about the `hasVolumeProperties`
        MxSubsurfaceData subsurfaceData = data.getSubsurfaceData();

        // We are not using the textured PF right now since we would need to store it in a volume stack.
        float3 g = 0;
        VolumeProperties result;
        if (any(subsurfaceData.w0 != float3(1.f)))
        {
            result.phaseFunction =
                DualHenyeyGreensteinPhaseFunction(subsurfaceData.meancosine0, subsurfaceData.meancosine1, subsurfaceData.w0);
            g = subsurfaceData.meancosine0 * subsurfaceData.w0 + subsurfaceData.meancosine1 * (1.f - subsurfaceData.w0);
        }
        else
        {
            result.phaseFunction = HenyeyGreensteinPhaseFunction(subsurfaceData.meancosine0[0]);
            g = subsurfaceData.meancosine0[0];
        }

        // Convert color to sigmaA. This is the van de Hulst solution with similarity relations applied.
        // See: Kulla and Conty: Revisiting Physically Based Shading at Imageworks, 2017
        const float3 c = subsurfaceData.color;
        const float3 s = 4.09712f + 4.20863f * c - sqrt(9.59217f + 41.6808f * c + 17.7126 * c * c);
        const float3 alpha = (1.f - s * s) / (1.f - g * s * s);

        // If the radius is 0, we would have infinitly asmall scatter distances.
        result.sigmaS = 1.f / max(subsurfaceData.radius, 0.00001f);
        result.sigmaA = select(alpha > 0.f, (1.f - alpha) * result.sigmaS / alpha, 0.f);

        return result;
    }

    Weights calculateWeights(float3 wiLocal[TBsdfCount], bool isFrontFacing)
    {
        Weights result;
        calculator.calculateWeights<TBsdfCount, TLayerCount, TLayeredMaterialData>(data, wiLocal, isFrontFacing, result);
        return result;
    }

    Weights calculatePlainWeights()
    {
        Weights result;
        calculator.calculatePlainWeights<TBsdfCount, TLayerCount, TLayeredMaterialData>(data, result);
        return result;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal[TBsdfCount];
        float3 woLocal[TBsdfCount];
        for (int i = 0; i < TBsdfCount; i++)
        {
            const float3 bsdfB = cross(data.getBsdfN(i), data.getBsdfT(i)) * sd.frame.getHandednessSign();
            wiLocal[i] = float3(dot(sd.V, data.getBsdfT(i)), dot(sd.V, bsdfB), dot(sd.V, data.getBsdfN(i)));
            woLocal[i] = float3(dot(wo, data.getBsdfT(i)), dot(wo, bsdfB), dot(wo, data.getBsdfN(i)));
        }

        Weights weightZeroA;
        Weights weightReverted;
        float3 normTransmission = float3(0.f);

        Weights weight = calculateWeights(wiLocal, data.getIsFrontFacing());

        // the following computations are only necessary if the have any transmission and if we are interested in the
        // lower hemisphere
        // test hemisphere with geometric normal
        // disabled for now
        // const bool reflect = dot(sd.faceN, sd.frame.N) > 0.0f ? dot(sd.V, sd.faceN) : dot(sd.V, -sd.faceN);
        // if (!reflect)
        {
            float3 expectedTransmission = 1.f;
            // TRANSMISSION TODO: figure out if we need t or t+at -> test with tinted simplebtdf
            for (int i = 0; i < TBsdfCount; i++)
            {
                const AlbedoContributions a = data.evalAlbedo(i, wiLocal[i], LobeType::Reflection);
                // we want to have r + ar + at
                expectedTransmission -= (1.f - a.transmission) * weight.bsdfWeights[i];
            }

            // safety clamp - if the albedos are incorrect, we need this
            expectedTransmission = clamp(expectedTransmission, 0.f, 1.f);

            if (any(expectedTransmission > 0.f))
            {
                weightReverted = calculateWeights(wiLocal, !data.getIsFrontFacing());
                weightZeroA = calculatePlainWeights();

                for (int i = 0; i < TBsdfCount; i++)
                {
                    if (data.getTransmits(i) != 0.f)
                    {
                        // see sample() for explanation
                        const AlbedoContributions a = data.evalAlbedo(i, wiLocal[i], LobeType::Reflection);

                        float3 realTransmission = select(
                            weightZeroA.bsdfWeights[i] > 0.f,
                            (weight.bsdfWeights[i] * weightReverted.bsdfWeights[i] * a.transmission) / weightZeroA.bsdfWeights[i],
                            0.f
                        );

                        normTransmission += select(expectedTransmission > 0.f, realTransmission / expectedTransmission, 0.f);
                    }
                }
            }
        }

        float3 eval = float3(0.f);
        for (int i = 0; i < TBsdfCount; i++)
        {
            // only look at respective hemisphere; alternatiley we could let the BSDF split up the contribution which
            // needs a change in the interfaces
            const bool reflect = wiLocal[i].z * woLocal[i].z > 0.0f;
            if (any(weight.bsdfWeights[i] > 0.f))
            {
                if (reflect)
                    eval += weight.bsdfWeights[i] * data.eval(i, wiLocal[i], woLocal[i], sg);
                else
                {
                    eval += select(
                        (normTransmission * weightZeroA.bsdfWeights[i]) > 0.f,
                        weight.bsdfWeights[i] * weightReverted.bsdfWeights[i] * data.getTransmits(i) /
                            (normTransmission * weightZeroA.bsdfWeights[i]) * data.eval(i, wiLocal[i], woLocal[i], sg),
                        0.f
                    );
                }
            }
        }

        if (any(isnan(eval)) || any(isinf(eval)))
            return 0.f;

        return eval;
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal[TBsdfCount];
        for (int i = 0; i < TBsdfCount; i++)
        {
            const float3 bsdfB = cross(data.getBsdfN(i), data.getBsdfT(i)) * sd.frame.getHandednessSign();
            wiLocal[i] = float3(dot(sd.V, data.getBsdfT(i)), dot(sd.V, bsdfB), dot(sd.V, data.getBsdfN(i)));
        }

        // weights top to bottom
        Weights weight = calculateWeights(wiLocal, data.getIsFrontFacing());

        // bottom up for transmission
        // PERFORMANCE TODO: Only calculate transmission parts if we have a transmissive material
        Weights weightReverted = calculateWeights(wiLocal, !data.getIsFrontFacing());

        // PERFORMANCE TODO: split up into reflection and transmission albedo -> do not sample transmission if it will
        // be blocked PERFORMANCE TODO: re-use albedo all over the place calculate selection weights to pick BSDF - bias
        // with absorption
        Weights selection;
        for (int i = 0; i < TBsdfCount; i++)
        {
            const AlbedoContributions a = data.evalAlbedo(i, wiLocal[i], LobeType::Reflection);
            selection.bsdfWeights[i] = luminance(weight.bsdfWeights[i] * (a.reflection + a.transmission * data.getTransmits(i)));
        }

        //  normalize weights for rand selection
        float normSelection = 0.f;
        for (int i = 0; i < TBsdfCount; i++)
            normSelection += selection.bsdfWeights[i].x;

        int i = 0;
        float sum = 0.f;
        const float rand = sampleNext1D(sg) * normSelection;
        for (i = 0; i < TBsdfCount - 1; ++i)
        {
            sum += selection.bsdfWeights[i].x;
            if (rand <= sum)
                break;
        }
        const int sampledLobe = i;

        // sample outgoing direction of selectd BSDF
        float3 woLocal = {};
        bool valid = data.sample(sampledLobe, wiLocal[sampledLobe], woLocal, result.pdf, result.weight, result.lobeType, sg);

        // apply weights
        result.weight *= weight.bsdfWeights[sampledLobe] * normSelection / selection.bsdfWeights[sampledLobe];

        const float3 bsdfB = cross(data.getBsdfN(sampledLobe), data.getBsdfT(sampledLobe)) * sd.frame.getHandednessSign();
        result.wo = data.getBsdfT(sampledLobe) * woLocal.x + bsdfB * woLocal.y + data.getBsdfN(sampledLobe) * woLocal.z;

        // transmission calculations:
        Weights weightZeroA = calculatePlainWeights();

        // PERFORMANCE TODO: this quantity should be used to shortcut transmission calculations -> move it to the top
        // and base the transmission calculations on it
        float3 expectedTransmission = 1.f;
        // TRANSMISSION TODO: figure out if we need t or t+at -> test with tinted simpllebtdf
        for (int i = 0; i < TBsdfCount; i++)
        {
            const AlbedoContributions a = data.evalAlbedo(i, wiLocal[i], LobeType::Reflection);
            // we want to have r + ar + at
            expectedTransmission -= (1.f - a.transmission) * weight.bsdfWeights[i];
        }

        // safety clamp - if the albedos are incorrect, we need this
        expectedTransmission = clamp(expectedTransmission, 0.f, 1.f);

        float3 normTransmission = float3(0.f);

        for (int i = 0; i < TBsdfCount; i++)
        {
            if (data.getTransmits(i) > 0.f)
            {
                // here we "convert" a layer into a fully un-mixed and un-weighted element; we know that R+a=T, we use
                // this T to reweight the transmission TRANSMISSION TODO: same as above, T or aT
                const AlbedoContributions a = data.evalAlbedo(i, wiLocal[i], LobeType::Reflection);

                float3 realTransmission = select(
                    weightZeroA.bsdfWeights[i] > 0.f,
                    (weight.bsdfWeights[i] * weightReverted.bsdfWeights[i] * a.transmission) / weightZeroA.bsdfWeights[i],
                    0.f
                );

                normTransmission += select(expectedTransmission > 0.f, realTransmission / expectedTransmission, 0.f);
            }
        }

        normTransmission = select(normTransmission < 1.f, 1.f, normTransmission);

        if ((result.lobeType & (uint)LobeType::Transmission) != 0)
        {
            // weights are frontWeights * backWeights * transmits and normalised by all transmissions
            // we divide out the second set of weights from the reverse traversal. E.g. if the BSDF has a weight of 0.5
            // applied on the way down, we don't want to reapply the same weight again on the way up
            result.weight *= select(
                normTransmission > 0.f,
                weightReverted.bsdfWeights[sampledLobe] * data.getTransmits(sampledLobe) /
                    (normTransmission * weightZeroA.bsdfWeights[sampledLobe]),
                0.f
            );
        }

        result.pdf = 0.f;
        for (int i = 0; i < TBsdfCount; i++)
        {
            if (any(weight.bsdfWeights[i] > 0.f) && normSelection > 0.f)
            {
                const float pdfBSDF = data.evalPdf(i, wiLocal[i], woLocal);
                // these two should be complimentary; one is for the upper hemisphere, one for the lower
                result.pdf += selection.bsdfWeights[i].x / normSelection * pdfBSDF;
                result.pdf += selection.bsdfWeights[i].x / normSelection * pdfBSDF * data.getTransmits(i);
            }
        }

        if (isnan(result.pdf) || isinf(result.pdf))
            return false;

        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal[TBsdfCount];
        float3 woLocal[TBsdfCount];
        for (int i = 0; i < TBsdfCount; i++)
        {
            const float3 bsdfB = cross(data.getBsdfN(i), data.getBsdfT(i)) * sd.frame.getHandednessSign();
            wiLocal[i] = float3(dot(sd.V, data.getBsdfT(i)), dot(sd.V, bsdfB), dot(sd.V, data.getBsdfN(i)));
            woLocal[i] = float3(dot(wo, data.getBsdfT(i)), dot(wo, bsdfB), dot(wo, data.getBsdfN(i)));
        }

        Weights weight = calculateWeights(wiLocal, data.getIsFrontFacing());

        Weights selection;
        for (int i = 0; i < TBsdfCount; i++)
        {
            const AlbedoContributions a = data.evalAlbedo(i, wiLocal[i], LobeType::Reflection);
            selection.bsdfWeights[i] = luminance(weight.bsdfWeights[i] * (a.reflection + a.transmission * data.getTransmits(i)));
        }

        float normSelection = 0.f;
        for (int i = 0; i < TBsdfCount; i++)
            normSelection += selection.bsdfWeights[i].x;

        const float wiDotN = dot(sd.V, sd.faceN);

        float pdf = 0.f;
        for (int i = 0; i < TBsdfCount; i++)
        {
            // pdfs are only based on the selection weights, not the layer weights
            if (any(weight.bsdfWeights[i] > 0.f))
            {
                const bool reflect = wiLocal[i].z * woLocal[i].z > 0.0f;
                if (reflect)
                    pdf += selection.bsdfWeights[i].x / normSelection * data.evalPdf(i, wiLocal[i], woLocal[i]);
                else
                    pdf += selection.bsdfWeights[i].x / normSelection * data.evalPdf(i, wiLocal[i], woLocal[i]) * data.getTransmits(i);
            }
        }

        if (isnan(pdf) || isinf(pdf))
            return 0.f;

        return pdf;
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        /// Properites of combined BSDFs; everything is view-dependent. Note that we use the roughness to split
        // diffuse and specular properties since BSDFs don't provide any other information
        // We ignore emission since there is no BSDF inferface to query.
        BSDFProperties p = {};

        float3 wiLocal[TBsdfCount];
        AlbedoContributions a[TBsdfCount];
        float3 transmissionWeights[TBsdfCount];
        float3 expectedTransmission = 1.f;
        for (int i = 0; i < TBsdfCount; i++)
        {
            const float3 bsdfB = cross(data.getBsdfN(i), data.getBsdfT(i)) * sd.frame.getHandednessSign();
            wiLocal[i] = float3(dot(sd.V, data.getBsdfT(i)), dot(sd.V, bsdfB), dot(sd.V, data.getBsdfN(i)));
            a[i] = data.evalAlbedo(i, wiLocal[i], LobeType::Reflection);
        }

        // Calculate reflection weights.
        Weights weight = calculateWeights(wiLocal, data.getIsFrontFacing());

        // Calculate transmission weights.
        for (int i = 0; i < TBsdfCount; i++)
            expectedTransmission -= (1.f - a[i].transmission) * weight.bsdfWeights[i];
        expectedTransmission = clamp(expectedTransmission, 0.f, 1.f);

        Weights weightReverted;
        Weights weightZeroA = calculatePlainWeights();
        float3 normTransmission = 0.f;
        float roughnessNorm = 0.0;
        if (any(expectedTransmission > 0.f))
        {
            weightReverted = calculateWeights(wiLocal, !data.getIsFrontFacing());

            for (int i = 0; i < TBsdfCount; i++)
            {
                if (data.getTransmits(i) != 0.f)
                {
                    float3 realTransmission = select(
                        weightZeroA.bsdfWeights[i] > 0.f,
                        (weight.bsdfWeights[i] * weightReverted.bsdfWeights[i] * a[i].transmission) / weightZeroA.bsdfWeights[i],
                        0.f
                    );

                    normTransmission += select(expectedTransmission > 0.f, realTransmission / expectedTransmission, 0.f);
                }
            }
        }

        for (int i = 0; i < TBsdfCount; i++)
        {
            // The diffuse roughness is 0.5.
            RoughnessInformation r = data.getRoughnessInformation(i, wiLocal[i]);
            const float contributionSplit = min(r.roughnessBSDFNotation[0] + r.roughnessBSDFNotation[1], 1.f);

            if (any(expectedTransmission > 0.f))
            {
                const float3 transmissionWeight = select(
                    (normTransmission * weightZeroA.bsdfWeights[i]) > 0.f,
                    weight.bsdfWeights[i] * weightReverted.bsdfWeights[i] * data.getTransmits(i) /
                        (normTransmission * weightZeroA.bsdfWeights[i]),
                    0.f
                );

                p.diffuseTransmissionAlbedo += a[i].transmission * transmissionWeight * contributionSplit;
                p.specularTransmissionAlbedo += a[i].transmission * transmissionWeight * (1.f - contributionSplit);
            }

            p.diffuseReflectionAlbedo += a[i].reflection * weight.bsdfWeights[i] * contributionSplit;
            p.specularReflectionAlbedo += a[i].reflection * weight.bsdfWeights[i] * (1.f - contributionSplit);

            // For the roughness, we take the unattenuated values. E.g. a red and a white diffuse should have the same roughness.
            // This means that the weights/albedos are essentially floats. The luminance() is more like a safety precaution.
            // Note that ideally we would want to re-calculate the weights without absorptionT and use the results for
            // weighting, but the overhead is probably not worth it. We take the average of both roughnesses.
            const float roughnessWeight = luminance(weightZeroA.bsdfWeights[i] * (a[i].reflection + a[i].absorptionR));
            p.roughness += roughnessWeight * (r.roughnessBSDFNotation[0] + r.roughnessBSDFNotation[1]) * 0.5f;
            roughnessNorm += roughnessWeight;

            // Calculate a representative guide normal as a weighted combination of the per-layer normals perturbed by normal mapping.
            // The guide normal is view-dependent and only accounts for the reflection component.
#ifdef UNATTENUATEDN
            p.guideNormal += luminance((a[i].reflection + a[i].absorptionR) * weight.bsdfWeights[i]) * data.getBsdfN(i);
#else
            p.guideNormal += luminance(a[i].reflection * weight.bsdfWeights[i]) * data.getBsdfN(i);
#endif
        }
        // Sanity clamps
        p.roughness = clamp(roughnessNorm > 0.0f ? p.roughness / roughnessNorm : 0.0, 0.f, 1.f);

        // We need to normalize because the weights will screw with the normals.
        p.guideNormal = normalize(p.guideNormal);
        p.specularReflectance = p.specularReflectionAlbedo;

        p.emission = 0.f;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        // GENERAL TODO: Return correct lobe types, i.e. go through all lobes. BSDFs won't support the infomation yet
        return (uint)LobeType::All;
    }

    ExtraBSDFProperties getExtraBSDFProperties(const ShadingData sd, const float3 wo)
    {
        ExtraBSDFProperties result;
        result.bsdfCount = TBsdfCount;

        float3 wiLocal[TBsdfCount];
        for (int i = 0; i < TBsdfCount; i++)
        {
            // TODO: Leave dummy BSDFs out
            // if (a[0] == 0.0 && a[1] == 1.0 && a[2] == 0.0 && a[3] == 0.0) ...
            // Compute wi to calculate weights below
            const float3 bsdfB = cross(data.getBsdfN(i), data.getBsdfT(i)) * sd.frame.getHandednessSign();
            wiLocal[i] = float3(dot(sd.V, data.getBsdfT(i)), dot(sd.V, bsdfB), dot(sd.V, data.getBsdfN(i)));

            // Express normal tangent, and bitangent in the smooth interpolated shading frame (sd).
            result.bsdfN[i] =
                float3(dot(data.getBsdfN(i), sd.frame.T), dot(data.getBsdfN(i), sd.frame.B), dot(data.getBsdfN(i), sd.frame.N));
            result.bsdfT[i] =
                float3(dot(data.getBsdfT(i), sd.frame.T), dot(data.getBsdfT(i), sd.frame.B), dot(data.getBsdfT(i), sd.frame.N));
            result.bsdfB[i] = float3(dot(bsdfB, sd.frame.T), dot(bsdfB, sd.frame.B), dot(bsdfB, sd.frame.N));
        }

        Weights weight = calculateWeights(wiLocal, data.getIsFrontFacing());

        for (int i = 0; i < TBsdfCount; i++)
        {
            AlbedoContributions a = data.getBsdf(i).evalAlbedo(wiLocal[i], LobeType::Reflection);
            result.bsdfAlbedo[i] = a.reflection;
            result.bsdfWeight[i] = weight.bsdfWeights[i];

            RoughnessInformation rough = data.getBsdf(i).getRoughnessInformation(wiLocal[i]);
            result.bsdfRoughness[i] = rough.roughnessBSDFNotation;
            result.bsdfScratch[i] = rough.scratch;
        }

        return result;
    }
}
