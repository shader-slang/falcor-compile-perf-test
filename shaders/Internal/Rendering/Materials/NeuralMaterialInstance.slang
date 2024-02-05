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
#include "Internal/Scene/Materials/NeuralMaterialTypes.slangh"
#include "Internal/Scene/Materials/NeuralMaterialEncodings.slangh"

__exported import Rendering.Materials.IMaterialInstance;
import Utils.Math.MathHelpers;
import Internal.Utils.Neural.Network;
import Scene.Scene;
import NeuralMaterialImportanceSampling;
import NeuralBSDFEvaluationNetwork; // This is a generated source defining the `BSDFEvaluationNetwork` type used below.

#ifndef NEURAL_MATERIAL_TYPE
#error Missing macro definition of NEURAL_MATERIAL_TYPE
#endif

#ifndef NEURAL_MATERIAL_INPUT_ENCODING
#error Missing macro definition of NEURAL_MATERIAL_INPUT_ENCODING
#endif
#ifndef NEURAL_MATERIAL_INPUT_NFRAMES
#error Missing macro definition of NEURAL_MATERIAL_INPUT_NFRAMES
#endif

#if NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME || NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME_RUSINKIEWICZ_CARTESIAN || NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME_RUSINKIEWICZ_SPHERICAL
// Input encoding is using a shading frame encoding network.
import NeuralShFrameNetwork; // This is a generated source defining the `ShFrameNetwork` type used below.
#ifndef NEURAL_MATERIAL_SHFRAME_PRECISION
#error Missing macro definition of NEURAL_MATERIAL_SHFRAME_PRECISION
#endif
#if NEURAL_MATERIAL_SHFRAME_PRECISION == NEURAL_MATERIAL_PRECISION_FP16
typedef float16_t ShFramePrecisionType;
#elif NEURAL_MATERIAL_SHFRAME_PRECISION == NEURAL_MATERIAL_PRECISION_FP32
typedef float ShFramePrecisionType;
#endif
#endif

#ifndef NEURAL_MATERIAL_PRECISION
#error Missing macro definition of NEURAL_MATERIAL_PRECISION
#endif
#if NEURAL_MATERIAL_PRECISION == NEURAL_MATERIAL_PRECISION_FP16
typedef float16_t PrecisionType;
#elif NEURAL_MATERIAL_PRECISION == NEURAL_MATERIAL_PRECISION_FP32
typedef float PrecisionType;
#endif

/**
 * Implementation of the material instance for the neural material.
 */
struct NeuralMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf; ///< Shading frame in world space.

    uint paramsByteOffsetEvaluation;          ///< Offset where evaluation network parameters are stored.
    uint paramsByteOffsetShFrame;             ///< Offset where shading frame network parameters are stored.
    uint paramsByteOffsetImportanceSampling1; ///< Offset where importance sampling network 1 parameters are stored.
    uint paramsByteOffsetImportanceSampling2; ///< Offset where importance sampling network 2 parameters are stored.

    float3 albedo; ///< Approximate albedo.
    float4 latent0;
    float4 latent1;

    float3 alpha;
    float2 slopeSpec;
    float2 slopeDiff;
    float weightSpec;

    const static bool kSquashInfNaNs = (NEURAL_MATERIAL_SQUASH_INF_NAN != 0);

    __init(
        const ShadingFrame sf,
        uint paramsByteOffsetEvaluation,
        uint paramsByteOffsetShFrame,
        uint paramsByteOffsetImportanceSampling1,
        uint paramsByteOffsetImportanceSampling2,
        float3 albedo,
        float4 latent0,
        float4 latent1,
        float3 alpha,
        float2 slopeSpec,
        float2 slopeDiff,
        float weightSpec
    )
    {
        this.sf = sf;
        this.paramsByteOffsetEvaluation = paramsByteOffsetEvaluation;
        this.paramsByteOffsetShFrame = paramsByteOffsetShFrame;
        this.paramsByteOffsetImportanceSampling1 = paramsByteOffsetImportanceSampling1;
        this.paramsByteOffsetImportanceSampling2 = paramsByteOffsetImportanceSampling2;
        this.albedo = albedo;
        this.latent0 = latent0;
        this.latent1 = latent1;
        this.alpha = alpha;
        this.slopeSpec = slopeSpec;
        this.slopeDiff = slopeDiff;
        this.weightSpec = weightSpec;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta)
            return float3(0.f);

        return evalLocal(sd, wiLocal, woLocal);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
#if !defined(NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING) && !defined(NEURAL_MATERIAL_USE_FLOW_IMPORTANCE_SAMPLING)
        useImportanceSampling = false;
#endif
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sampleLocal(wiLocal, sampleNext2D(sg), result.pdf, useImportanceSampling);

        result.wo = sf.fromLocal(woLocal);
        result.weight = evalLocal(sd, wiLocal, woLocal) / result.pdf;
        result.lobeType = (uint)LobeType::DiffuseReflection;

        // We do the hemisphere checks after evaluation of the weight above to increase warp utilization.
        // This is an important performance optimization when Tensor Cores are enabled.
        if (min(wiLocal.z, woLocal.z) < kMinCosTheta || result.pdf < 1e-3f)
            return false;

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
#if !defined(NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING) && !defined(NEURAL_MATERIAL_USE_FLOW_IMPORTANCE_SAMPLING)
        useImportanceSampling = false;
#endif
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (min(wiLocal.z, woLocal.z) < kMinCosTheta)
            return 0.f;

        return max(1e-3f, evalPdfLocal(wiLocal, woLocal, useImportanceSampling));
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
#ifdef NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING
        // Use weighted diffuse/specular normal as guide normal.
        float3 nd = normalize(float3(-slopeDiff.x, -slopeDiff.y, 1.f));
        float3 ns = normalize(float3(-slopeSpec.x, -slopeSpec.y, 1.f));
        float3 n = normalize(weightSpec * ns + (1.f - weightSpec) * nd);
        p.guideNormal = sf.fromLocal(n);
#endif

        // Compute approximation of the perceived roughness. For now assume a medium roughness.
        p.roughness = 0.5f;
#ifdef NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING
        // Use average alpha as proxy for isotropic roughness.
        float isoAlpha = (alpha.x + alpha.y) * 0.5f;
        p.roughness = sqrt(isoAlpha);
#endif

        // Return approximation of the albedos.
        // The BSDF reports only a diffuse reflection lobe, so leaving the other albedos as zero.
        p.diffuseReflectionAlbedo = albedo;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        // Report a single diffuse reflection lobe if we have no further information.
        return (uint)LobeType::DiffuseReflection;
    }

    // Internal helpers

    /**
     * Evaluate the BRDF in the local frame.
     * The vectors are assumed to be in the upper hemisphere.
     * @param[in] sd Shading data.
     * @param[in] wi Incident direction in the local frame.
     * @param[in] wo Outgoing direction in the local frame.
     * @return f(wi, wo) * wo.z
     */
    float3 evalLocal(const ShadingData sd, const float3 wi, const float3 wo)
    {
        // We start by preparing the inputs for the evaluation network.
        //
        // There are currently two mechanisms for deciding how inputs should be encoded:
        // 1. The neural material type selects between different hard-coded material types.
        //    This is mostly a legacy thing and will be removed.
        // 2. The input encoding enum selects between different hard-coded input encodings.
        //    This is currently only used for the 'WithShFrameExtended' neural material type.
        //
        const uint kinputWidth = NEURAL_MATERIAL_INPUT_WIDTH;
        const uint kOutputWidth = NEURAL_MATERIAL_OUTPUT_WIDTH;

        // ---------------------------------------------------------------------------------------
        // Legacy neural material types.
        // These used fixed input encodings for each type.
        // ---------------------------------------------------------------------------------------
#if NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_LEGACY_MERL
        // Input encoding: Rusinkiewicz parameterization.
        // TODO: remove some of the trig, we can do the above using double angle identities etc.
        // phi_h is thrown away (assume isotropic material)
        float3 h = normalize(wi + wo);
        float theta_h, phi_h;
        float theta_d, phi_d;
        wi_h_to_rusinkiewicz(wi, h, theta_h, phi_h, theta_d, phi_d);

        // Use a double angle periodic encoding to enforce reciprocity.
        float s, t;
        sincos(2.f * phi_d, s, t);

        // Prepare network input features.
        // The input width was padded to 8 (tcnn min size).
        // TODO: Use frequency encoding to pad.
        PrecisionType input[8] = { theta_h, theta_d, s, t, 1.f, 1.f, 1.f, 1.f };

#elif NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_BASIC
        PrecisionType input[8] = {
            sd.uv.x,
            sd.uv.y,
            wi.x,
            wi.y,
            wi.z,
            wo.x,
            wo.y,
            wo.z,
        };

#elif NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_LATENT_TEXTURES || NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_WITH_SHFRAME
        PrecisionType input[16] = {
            sd.uv.x,
            sd.uv.y,
            wi.x,
            wi.y,
            wi.z,
            wo.x,
            wo.y,
            wo.z,
            latent0.x,
            latent0.y,
            latent0.z,
            latent0.w,
            latent1.x,
            latent1.y,
            latent1.z,
            latent1.w,
        };

        // ---------------------------------------------------------------------------------------
        // The most general neural material type that supports configurable input encodings.
        // We prepare the inputs differently based on the NEURAL_MATERIAL_INPUT_ENCODING define.
        // ---------------------------------------------------------------------------------------
#elif NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_WITH_SHFRAME_EXTENDED
#if NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_IDENTITY
        // The 'Identity' encoding passes in (wi, wo, latents) unmodified.
        PrecisionType input[14] = {
            wi.x,
            wi.y,
            wi.z,
            wo.x,
            wo.y,
            wo.z,
            latent0.x,
            latent0.y,
            latent0.z,
            latent0.w,
            latent1.x,
            latent1.y,
            latent1.z,
            latent1.w,
        };

#elif NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME || NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME_RUSINKIEWICZ_CARTESIAN || NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME_RUSINKIEWICZ_SPHERICAL
        // These encodings use a small network to generate shading frames from the latent code.
        // The NEURAL_MATERIAL_INPUT_NFRAMES define determines number of generated shading frames.
        const ShFramePrecisionType latents[8] = {
            latent0.x,
            latent0.y,
            latent0.z,
            latent0.w,
            latent1.x,
            latent1.y,
            latent1.z,
            latent1.w,
        };
        ShFramePrecisionType shframes[NEURAL_MATERIAL_INPUT_NFRAMES * 6] = {};
        {
            // Execute shading frame network to generate predicted shading frames.
            // The input is the texture latent codes.
            // The output is an array of predicted normals and tangents: [n[0], n[1], .., n[N-1], t[0], t[1], .., t[N-1]].
            // The shading frame network currently always uses fp32 precision.
#if NEURAL_MATERIAL_SHFRAME_PRECISION == NEURAL_MATERIAL_PRECISION_FP16
            StructuredBuffer<NetworkParamStorageFP16> params = gScene.materials.networkParamsFP16;
#elif NEURAL_MATERIAL_SHFRAME_PRECISION == NEURAL_MATERIAL_PRECISION_FP32
            StructuredBuffer<NetworkParamStorageFP32> params = gScene.materials.networkParamsFP32;
#endif
            ShFrameNetwork mlp;
            mlp.eval(
                params,
                gScene.materials.networkWeightsTin,
                gScene.materials.networkBiasTin,
                gScene.materials.networkParamsNVCoopVec,
                latents,
                shframes,
                paramsByteOffsetShFrame
            );
        }

        // Transform (wi, wo) into the predicted shading frames.
        // The result is stored in the 'shframes' array in the layout: [wi[0], wi[1], .., wi[N-1], wo[0], wo[1], .., wo[N-1]].
        [unroll]
        for (uint i = 0; i < NEURAL_MATERIAL_INPUT_NFRAMES; i++)
        {
            float3 N = float3(shframes[3 * i + 0], shframes[3 * i + 1], shframes[3 * i + 2]);
            float3 T = float3(
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 0],
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 1],
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 2]
            );
            // Add default normal/tangent applied during training.
            N += float3(0, 0, 1);
            T += float3(1, 0, 0);

            // Setup shading frame exactly as done in python.
            // Note that the frame is not orthogonalized (T is not necessarily orthogonal to N).
            ShadingFrame sf;
            sf.N = normalize(N);
            sf.T = normalize(T);
            sf.B = cross(sf.N, sf.T);

            // Transform (wi, wo) into the predicted frame and store out result.
            float3 wiPred = sf.toLocal(wi);
            float3 woPred = sf.toLocal(wo);

            shframes[3 * i + 0] = wiPred.x;
            shframes[3 * i + 1] = wiPred.y;
            shframes[3 * i + 2] = wiPred.z;
            shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 0] = woPred.x;
            shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 1] = woPred.y;
            shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 2] = woPred.z;
        }

#if NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME
        // The 'ShFrame' encoding passes in the array of (wi, wo) transformed into the predicted frames unmodified, followed by the original
        // latents. Quantize to the precision used by the evaluation network.
        PrecisionType input[NEURAL_MATERIAL_INPUT_NFRAMES * 6 + 8];

        [unroll]
        for (uint i = 0; i < NEURAL_MATERIAL_INPUT_NFRAMES * 6; i++)
            input[i] = shframes[i];
        [unroll]
        for (uint i = 0; i < 8; i++)
            input[NEURAL_MATERIAL_INPUT_NFRAMES * 6 + i] = latents[i];

#elif NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME_RUSINKIEWICZ_CARTESIAN
        // Each of the (wi, wo) rotated into the predicted shading frames is transformed to (h, d) vectors
        // using Rusinkiewicz parameterization in Cartesian coordinates.
        // The output is in the layout [h[0], h[1], .., h[N-1], d[0], d[1], .., d[N-1]] where each element is 3-wide vector.
        PrecisionType input[NEURAL_MATERIAL_INPUT_NFRAMES * 6 + 8];

        [unroll]
        for (uint i = 0; i < NEURAL_MATERIAL_INPUT_NFRAMES; i++)
        {
            float3 wi = float3(shframes[3 * i + 0], shframes[3 * i + 1], shframes[3 * i + 2]);
            float3 wo = float3(
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 0],
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 1],
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 2]
            );

            float3 h, d;
            computeRusinkiewiczCartesian(wi, wo, h, d);

            // Store result directly into quantized 'input' array we'll feed to the evaluation network.
            input[3 * i + 0] = h.x;
            input[3 * i + 1] = h.y;
            input[3 * i + 2] = h.z;
            input[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 0] = d.x;
            input[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 1] = d.y;
            input[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 2] = d.z;
        }
        [unroll]
        for (uint i = 0; i < 8; i++)
            input[NEURAL_MATERIAL_INPUT_NFRAMES * 6 + i] = latents[i];

#elif NEURAL_MATERIAL_INPUT_ENCODING == NEURAL_MATERIAL_INPUT_ENCODING_SHFRAME_RUSINKIEWICZ_SPHERICAL
        // Each of the (wi, wo) rotated into the predicted shading frames is transformed to (h, d) vectors
        // using Rusinkiewicz parameterization in spherical coordinates.
        // The output is in the layout [h[0], h[1], .., h[N-1], d[0], d[1], .., d[N-1]] where each element is 4-wide vector.
        PrecisionType input[NEURAL_MATERIAL_INPUT_NFRAMES * 8 + 8];

        [unroll]
        for (uint i = 0; i < NEURAL_MATERIAL_INPUT_NFRAMES; i++)
        {
            float3 wi = float3(shframes[3 * i + 0], shframes[3 * i + 1], shframes[3 * i + 2]);
            float3 wo = float3(
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 0],
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 1],
                shframes[3 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 2]
            );

            float4 h_sph, d_sph;
            computeRusinkiewiczSpherical(wi, wo, h_sph, d_sph);

            // Store result directly into quantized 'input' array we'll feed to the evaluation network.
            input[4 * i + 0] = h_sph.x;
            input[4 * i + 1] = h_sph.y;
            input[4 * i + 2] = h_sph.z;
            input[4 * i + 3] = h_sph.w;
            input[4 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 0] = d_sph.x;
            input[4 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 1] = d_sph.y;
            input[4 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 2] = d_sph.z;
            input[4 * (NEURAL_MATERIAL_INPUT_NFRAMES + i) + 3] = d_sph.w;
        }
        [unroll]
        for (uint i = 0; i < 8; i++)
            input[NEURAL_MATERIAL_INPUT_NFRAMES * 8 + i] = latents[i];

#else
#error Unexpected NEURAL_MATERIAL_INPUT_ENCODING value
#endif

#else
#error Unsupported NEURAL_MATERIAL_INPUT_ENCODING
#endif // NEURAL_MATERIAL_INPUT_ENCODING

#else
#error Unsupported neural material type
#endif

#if NEURAL_MATERIAL_PRECISION == NEURAL_MATERIAL_PRECISION_FP16
        StructuredBuffer<NetworkParamStorageFP16> params = gScene.materials.networkParamsFP16;
#elif NEURAL_MATERIAL_PRECISION == NEURAL_MATERIAL_PRECISION_FP32
        StructuredBuffer<NetworkParamStorageFP32> params = gScene.materials.networkParamsFP32;
#endif

        PrecisionType output[NEURAL_MATERIAL_OUTPUT_WIDTH] = {};

        // Evaluate MLP.
        // The output is the BRDF evaluation including the cosine term for the outgoing direction.
        // The network code is generated and compiled into the program as a separate module imported above.
        BSDFEvaluationNetwork mlp;
        mlp.eval(
            params,
            gScene.materials.networkWeightsTin,
            gScene.materials.networkBiasTin,
            gScene.materials.networkParamsNVCoopVec,
            input,
            output,
            paramsByteOffsetEvaluation
        );

        float3 f = float3(output[0], output[1], output[2]);

        // Fixup result by squashing Inf/NaNs that may be produced by the network to zeros.
        // Also clamp to avoid negative output in case the output activations function can produce negative results.
        if (kSquashInfNaNs)
        {
            f = max(f, float3(0.f));
            if (any(isinf(f)) || any(isnan(f)))
                f = float3(0.f);
        }

#if NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_LEGACY_MERL
        f *= wo.z * M_1_PI; // WAR: Alex' data was trained on MERL * pi and didn't include the cosine term.
#endif
        return f;
    }

    float3 sampleLocal(float3 wi, float2 u, out float pdf, bool useImportanceSampling)
    {
        if (useImportanceSampling)
        {
#if defined(NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING)
            return sampleNeuralBSDFProxy(alpha, slopeSpec, slopeDiff, weightSpec, wi, u, pdf);
#elif defined(NEURAL_MATERIAL_USE_FLOW_IMPORTANCE_SAMPLING)
            return sampleNeuralBSDFFlow(
                paramsByteOffsetImportanceSampling1, paramsByteOffsetImportanceSampling2, latent0, latent1, wi, u, pdf
            );
#else
            pdf = 0.f;
            return float3(0.f);
#endif
        }
        else
        {
            return sample_cosine_hemisphere_concentric(u, pdf);
        }
    }

    float evalPdfLocal(float3 wi, float3 wo, bool useImportanceSampling)
    {
        if (useImportanceSampling)
        {
#if defined(NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING)
            return evalPdfNeuralBSDFProxy(alpha, slopeSpec, slopeDiff, weightSpec, wi, wo);
#elif defined(NEURAL_MATERIAL_USE_FLOW_IMPORTANCE_SAMPLING)
            return evalPdfNeuralBSDFFlow(
                paramsByteOffsetImportanceSampling1, paramsByteOffsetImportanceSampling2, latent0, latent1, wi, wo
            );
#else
            return 0.f;
#endif
        }
        else
        {
            return wo.z * M_1_PI;
        }
    }

    // TODO: Move these helpers into MathUtils.slang and unify the conventions.
    // We add unit tests and make sure there is corresponding helpers in Python that are also tested.

    void cartesian_to_spherical(float3 v, out float theta, out float phi)
    {
        theta = acos(clamp(v.z, -1.f, 1.f));
        phi = atan2(v.y, v.x);
    }

    void wi_h_to_rusinkiewicz(float3 wi, float3 h, out float theta_h, out float phi_h, out float theta_d, out float phi_d)
    {
        cartesian_to_spherical(h, theta_h, phi_h);
        float3 temp = rotate_z(wi, -phi_h);
        float3 d = rotate_y(temp, -theta_h);
        cartesian_to_spherical(d, theta_d, phi_d);
    }

    // Functions that should match with python side.

    void cartesianToSpherical(const float3 v, out float theta, out float phi, const float eps = 1e-6f)
    {
        float x = abs(v.x) < eps ? sign(v.x) * eps : v.x;
        float y = abs(v.y) < eps ? sign(v.y) * eps : v.y;
        float z = clamp(v.z, -1.f + eps, 1.f - eps);
        theta = acos(z);
        phi = atan2(y, x) + M_PI;
    }

    void computeRusinkiewiczCartesian(const float3 wi, const float3 wo, out float3 h, out float3 d)
    {
        h = normalize(wi + wo);
        float theta_h, phi_h;
        cartesianToSpherical(h, theta_h, phi_h);
        d = rotate_y(rotate_z(wi, -phi_h), -theta_h);
    }

    void computeRusinkiewiczSpherical(const float3 wi, const float3 wo, out float4 h_sph, out float4 d_sph)
    {
        float3 h = normalize(wi + wo);
        float theta_h, phi_h;
        cartesianToSpherical(h, theta_h, phi_h);
        float3 d = rotate_y(rotate_z(wi, -phi_h), -theta_h);

        float theta_d, phi_d;
        cartesianToSpherical(d, theta_d, phi_d);

        h_sph = { sin(theta_h), cos(theta_h), sin(phi_h), cos(phi_h) };
        d_sph = { sin(theta_d), cos(theta_d), sin(2.f * phi_d), cos(2.f * phi_d) };
    }

    ExtraBSDFProperties getExtraBSDFProperties(const ShadingData sd, const float3 wo)
    {
        ExtraBSDFProperties result;
        result.bsdfCount = 0;
        return result;
    }
};
