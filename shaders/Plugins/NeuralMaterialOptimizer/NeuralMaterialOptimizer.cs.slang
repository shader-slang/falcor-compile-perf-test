/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Helpers;
import Encoder;
import ShadingFrameDecoder;
import EvaluationDecoder;
import SamplingDecoder;
import LatentTexture;
import LatentHashGrid;
import LatentDictionary;
import LatentSphericalHarmonics;
import LearnedSampler;
import Plugins.MaterialDataGenerator.MaterialDataGenerator;

#include "Utils/NVAPI.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "DebugInfo.slangh"

#define SKIP_INVALID_GRADIENTS

static const uint32_t kNumShadingFrames = NUM_SHADING_FRAMES;
static const uint32_t kNumDirectionalInputs = NUM_DIRECTIONAL_INPUTS;

cbuffer CB
{
    // For evaluating the model.
    uint evalBufferSize;          // Number of elements in the buffers below.
    uint sampleBufferContent;     // Flags describing valid data in the next three buffers.
    uint fetchLatentsFromTexture; // Whether latents are already availble (e.g. after a checkpoint), or need to be fetched via the encoder.
    StructuredBuffer<float2> uvBuffer;              // Texture coordinates to evaluate.
    StructuredBuffer<float3> wiBuffer;              // Incident directions to evaluate.
    StructuredBuffer<float3> woBuffer;              // Outgoing directions to evaluate.
    RWStructuredBuffer<float3> targetEvalBuffer;    // RGB output of the target eval model;
    RWStructuredBuffer<float3> learnedEvalBuffer;   // RGB output of the learned eval model.
    RWStructuredBuffer<float> learnedPdfBuffer;     // RGB output of the learned PDF model.
    RWStructuredBuffer<float3> targetNormalBuffer;  // XYZ output of the target normal(s).
    RWStructuredBuffer<float3> learnedNormalBuffer; // XYZ output of the learned normal(s).
#ifdef USE_TRAINING_DATA_BUFFER
    StructuredBuffer<MaterialSample> sampleBuffer; // Buffer holding training records.
#endif

    // For optimization.
    RWTexture2D<float> evalLossTexture;     // Texture holding the eval model loss for the full batch.
    RWTexture2D<float> samplingLossTexture; // Texture holding the sampling model loss for the full batch.
    uint iteration;                         // Current iteration of the optimization. Starts at zero.
    float evaluationLossScale;              // Loss scale for the evaluation model (used during FP16 precision training).
    float samplingLossScale;                // Loss scale for the sampling model (used during FP16 precision training).
    float l2ParameterRegularization;        // Scale factor for L2 parameter regularization.

    // For debugging.
    RWStructuredBuffer<DebugInfo> debugBuffer; // Holds a `DebugInfo` struct for each entry in the batch.

    // For direct latent texture optimization or encoding into the textures.
    uint activeTile;  // Selected UDIM tile to operate on.
    uint activeLevel; // Selected level to operate on.
    uint instance;    // Current instance for optimization pass.
}

struct AdamParameters
{
    // Adam parameters
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
};
ParameterBlock<AdamParameters> gAdamParams;

ParameterBlock<MaterialDataGenerator> gMaterialDataGenerator;
ParameterBlock<Encoder> gEncoder;
ParameterBlock<ShadingFrameDecoder> gShadingFrameDecoder;
ParameterBlock<EvaluationDecoder> gEvaluationDecoder;
ParameterBlock<LatentTexture> gLatentTexture;
ParameterBlock<LatentHashGrid> gLatentHashGrid;
ParameterBlock<LatentDictionary> gLatentDictionary;
ParameterBlock<LatentSphericalHarmonics> gLatentSphericalHarmonics;

float nanCheck(float x, uint batchIdx)
{
    return x;
}

[BackwardDerivativeOf(nanCheck)]
void bwd__nanCheck(inout DifferentialPair<float> x, uint batchIdx, float dOut)
{
    debugBuffer[batchIdx].nanCheck = dOut;
    x = DifferentialPair<float>(x.p, dOut);
}

void createEncoderInput(MaterialSample ms, out float input[Encoder.kNumInputs])
{
    const uint k = 14;
    for (uint i = 0; i < NUM_EXPORTED_BSDFS; i++)
    {
        input[(i * k) + 0] = ms.normal[i][0];
        input[(i * k) + 1] = ms.normal[i][1];
        input[(i * k) + 2] = ms.normal[i][2];
        input[(i * k) + 3] = ms.tangent[i][0];
        input[(i * k) + 4] = ms.tangent[i][1];
        input[(i * k) + 5] = ms.tangent[i][2];
        input[(i * k) + 6] = ms.albedo[i][0];
        input[(i * k) + 7] = ms.albedo[i][1];
        input[(i * k) + 8] = ms.albedo[i][2];
        input[(i * k) + 9] = ms.roughness[i][0];
        input[(i * k) + 10] = ms.roughness[i][1];
        input[(i * k) + 11] = ms.bsdfWeight[i][0];
        input[(i * k) + 12] = ms.bsdfWeight[i][1];
        input[(i * k) + 13] = ms.bsdfWeight[i][2];
    }
    input[k * NUM_EXPORTED_BSDFS] = ms.level;
}

[BackwardDifferentiable]
void fetchLatentCode(
    MaterialSample ms,
    no_diff float sample,
    uint batchIdx,
    uint warpIdx,
    out float latentCode[LatentTexture.kNumLatentDims]
)
{
#if defined(USE_LATENT_HASH_GRID)
    gLatentHashGrid.getCodeBilinear(ms.uv, latentCode);
#elif defined(USE_LATENT_ENCODER)
    // Evaluate latents through encoder.
    float encInput[Encoder.kNumInputs];
    createEncoderInput(ms, encInput);
    gEncoder.eval(encInput, latentCode, batchIdx, warpIdx);
#else
    // Get latents from explicit texture, in case of direct texture optimization.
    gLatentTexture.getCodeBilinearStochastic(ms.uv, ms.level, sample, latentCode);
#endif

#ifdef FREEZE_LATENT_TEXTURE
    // Optionally freeze the latents in place and focus on finetuning the remainder of the model.
    [ForceUnroll]
    for (int i = 0; i < LatentTexture.kNumLatentDims; i++)
    {
        latentCode[i] = detach(latentCode[i]);
    }
#endif
}

[BackwardDifferentiable]
void evalModelAndLoss(uint3 rngSeed, uint batchIdx, uint warpIdx, uint3 threadIdx, uint2 groupSize, out float loss[2])
{
    // Generate a single sample from the material data generator. Not contrained to any uv/wi/wo.
    float2 uvFixed;
    float3 wiFixed;
    float3 woFixed;
    SampleBufferContentDesc fixedInputs = SampleBufferContentDesc::None;

#ifdef USE_TRAINING_DATA_BUFFER
    uint msIdx = (threadIdx.z * groupSize.x * groupSize.y) + threadIdx.y * groupSize.x + threadIdx.x;
    MaterialSample ms = sampleBuffer[msIdx];
#else
    MaterialSample ms = no_diff gMaterialDataGenerator.generateMaterialSample(threadIdx, groupSize, fixedInputs, uvFixed, wiFixed, woFixed);
#endif

    // Random number generator.
    SampleGenerator sg = SampleGenerator(rngSeed.xy, rngSeed.z);

    // Get the latent code.
    float latentCode[LatentTexture.kNumLatentDims];
    fetchLatentCode(ms, sampleNext1D(sg), batchIdx, warpIdx, latentCode);

    loss[0] = 0.f; // Evaluation model loss.
    loss[1] = 0.f; // Sampling model loss.

    // Skipping invalid samples leads to incorrect gradients when
    // using TIN, instead we use ms.valid to scale the loss. Since
    // ms.valid==0.f for invalid samples, this amounts to skipping.
    // if (ms.valid)
    {
        // Run evaluation model and compare against the target value.
        float3 prediction = evaluationDecoder(ms.wi, ms.wo, latentCode, batchIdx, warpIdx);
        float3 reference = ms.weight;

        // LogL1 loss computation.
        float3 logReference = log(1.f + reference);
        float3 logPrediction = log(1.f + prediction);
        float3 diff = abs(logReference - logPrediction);
        loss[0] = ms.valid * (diff[0] + diff[1] + diff[2]) / 3.f;
    }

    // Train importance sampling, if enabled.
    {
        // Run sampling decoder to infer proxy parameters from the detached latent code.
        var params = LearnedSampler.decodeParams(ms.wi, latentCode, batchIdx, warpIdx);

        // Sample outgoing direction (without tracking derivatives).
        float3 wo = no_diff LearnedSampler.sample(ms.wi, sampleNext3D(sg), params);

        // Evaluate the learned PDF for the sampled direction.
        float pdf = LearnedSampler.pdf(ms.wi, wo, params);

        // Evaluate the learned evaluation network for the sampled direction.
        float3 bsdf = evaluationDecoder(ms.wi, wo, detach(latentCode), batchIdx, warpIdx);
        float reference = luminance(detach(bsdf));

        // Compute a Monte Carlo approximation of the KL-divergence loss.
        loss[1] = -detach(reference) * log(pdf) / detach(pdf);
        if (ms.wi.z <= 0.f || wo.z <= 0.f)
            loss[1] = 0.f;
    }

    uint2 lossIdx = uint2(batchIdx % 4096, batchIdx / 4096);
    evalLossTexture[lossIdx] = detach(loss[0]);
    samplingLossTexture[lossIdx] = detach(loss[1]);
}

interface IParameterization
{
    static void transform(inout float3 wi, inout float3 wo);
}

struct IdentityParameterization : IParameterization
{
    static void transform(inout float3 wi, inout float3 wo) { return; }
}

struct RusinkiewiczParameterization : IParameterization
{
    static void transform(inout float3 wi, inout float3 wo)
    {
        // retrieve half vector
        float3 half = normalize(wi + wo);
        float2 half_sph = cartesian_to_spherical_rad(half);
        // retrieve diff vector
        float3 temp = rotate_z(wi, -half_sph.y);
        float3 diff = rotate_y(temp, -half_sph.x);
        // replace (wo, wi) with (half, diff)
        wi = half;
        wo = diff;
    }
}

#ifdef USE_RUSINKIEWICZ_PARAMETERIZATION
typedef RusinkiewiczParameterization Parameterization;
#else
typedef IdentityParameterization Parameterization;
#endif

interface IShadingFrame
{
    [BackwardDifferentiable]
    static void toLocal(
        float3 wi,
        float3 wo,
        float latentCode[LatentTexture.kNumLatentDims],
        uint batchIdx,
        uint warpIdx,
        out float3 wiLocal[kNumShadingFrames],
        out float3 woLocal[kNumShadingFrames]
    );
}

struct IdentityShadingFrame : IShadingFrame
{
    [BackwardDifferentiable]
    static void toLocal(
        float3 wi,
        float3 wo,
        float latentCode[LatentTexture.kNumLatentDims],
        uint batchIdx,
        uint warpIdx,
        out float3 wiLocal[kNumShadingFrames],
        out float3 woLocal[kNumShadingFrames]
    )
    {
        [ForceUnroll]
        for (int i = 0; i < kNumShadingFrames; i++)
        {
            wiLocal[i] = wi;
            woLocal[i] = wo;
        }
    }
}

struct LearnedShadingFrame : IShadingFrame
{
    [BackwardDifferentiable]
    static void toLocal(
        float3 wi,
        float3 wo,
        float latentCode[LatentTexture.kNumLatentDims],
        uint batchIdx,
        uint warpIdx,
        out float3 wiLocal[kNumShadingFrames],
        out float3 woLocal[kNumShadingFrames]
    )
    {
        // Run shading frame decoder.
        float shframes[6 * kNumShadingFrames];
#ifdef RUN_SHADING_FRAME_DECODER
        gShadingFrameDecoder.eval(latentCode, shframes, batchIdx, warpIdx);
#endif

        [ForceUnroll]
        for (int i = 0; i < kNumShadingFrames; i++)
        {
            int i1 = i + 0;                 // i1 indexes the first half of the buffer
            int i2 = i + kNumShadingFrames; // i2 indexes the second half of the buffer
            // Extract normal and tangent.
            float3 N = float3(shframes[3 * i1 + 0], shframes[3 * i1 + 1], shframes[3 * i1 + 2]);
            float3 T = float3(shframes[3 * i2 + 0], shframes[3 * i2 + 1], shframes[3 * i2 + 2]);

            // Add standard normal/tangent on top.
            N += float3(0.f, 0.f, 1.f);
            T += float3(1.f, 0.f, 0.f);

            // Normalize and compute bitangent.
            N = normalize(N);
            T = normalize(T);
            float3 B = cross(N, T);

            // Apply the rotations.
            wiLocal[i] = float3(dot(wi, T), dot(wi, B), dot(wi, N));
            woLocal[i] = float3(dot(wo, T), dot(wo, B), dot(wo, N));

#ifdef STORE_NORMALS
            learnedNormalBuffer[batchIdx * kNumShadingFrames + i] = detach(N);
#endif
        }
    }
}

#ifdef RUN_SHADING_FRAME_DECODER
typedef LearnedShadingFrame ShadingFrame;
#else
typedef IdentityShadingFrame ShadingFrame;
#endif

interface IEncoding
{
    static const uint offset;
    [BackwardDifferentiable]
    static void prepareInput(
        float3 wiLocal[kNumShadingFrames],
        float3 woLocal[kNumShadingFrames],
        out float input[EvaluationDecoder.kNumInputs]
    );
}

struct IdentityEncoding : IEncoding
{
    static const uint offset = kNumShadingFrames * 2 * 3;
    [BackwardDifferentiable]
    static void prepareInput(
        float3 wiLocal[kNumShadingFrames],
        float3 woLocal[kNumShadingFrames],
        out float input[EvaluationDecoder.kNumInputs]
    )
    {
        [ForceUnroll]
        for (uint i = 0; i < kNumShadingFrames; i++)
        {
            int i1 = i + 0;                 // i1 indexes the first half of the buffer
            int i2 = i + kNumShadingFrames; // i2 indexes the second half of the buffer
            input[3 * i1 + 0] = wiLocal[i][0];
            input[3 * i1 + 1] = wiLocal[i][1];
            input[3 * i1 + 2] = wiLocal[i][2];
            input[3 * i2 + 0] = woLocal[i][0];
            input[3 * i2 + 1] = woLocal[i][1];
            input[3 * i2 + 2] = woLocal[i][2];
        }
    }
}

struct LatentDictionaryEncoding : IEncoding
{
    static const uint offset = kNumShadingFrames * 2 * gLatentDictionary.kNumLatentDims;
    static uint wiInstance = WI_INSTANCE; // Index of encoding instanced used for wi directions.
    static uint woInstance = WO_INSTANCE; // Index of encoding instanced used for wo directions.

    [BackwardDifferentiable]
    static void prepareInput(
        float3 wiLocal[kNumShadingFrames],
        float3 woLocal[kNumShadingFrames],
        out float input[EvaluationDecoder.kNumInputs]
    )
    {
        [ForceUnroll]
        for (uint i = 0; i < kNumShadingFrames; i++)
        {
            uint i1 = i + 0;                 // Used to index the first half of the buffer
            uint i2 = i + kNumShadingFrames; // Used to index the second half of the buffer
            float wiEncoded[gLatentDictionary.kNumLatentDims];
            float woEncoded[gLatentDictionary.kNumLatentDims];
            gLatentDictionary.getCodeBilinear(wiLocal[i], wiEncoded, wiInstance);
            gLatentDictionary.getCodeBilinear(woLocal[i], woEncoded, woInstance);
            [ForceUnroll]
            for (uint j = 0; j < gLatentDictionary.kNumLatentDims; j++)
            {
                input[gLatentDictionary.kNumLatentDims * i1 + j] = wiEncoded[j];
                input[gLatentDictionary.kNumLatentDims * i2 + j] = woEncoded[j];
            }
        }
    }
}

struct LatentBiplaneEncoding : IEncoding
{
    static const uint offset = kNumShadingFrames * (3 + gLatentDictionary.kNumLatentDims);
    static uint wiInstance = WI_INSTANCE; // Index of encoding instanced used for wo directions.

    [BackwardDifferentiable]
    static void prepareInput(
        float3 wiLocal[kNumShadingFrames],
        float3 woLocal[kNumShadingFrames],
        out float input[EvaluationDecoder.kNumInputs]
    )
    {
        [ForceUnroll]
        for (uint i = 0; i < kNumShadingFrames; i++)
        {
            uint woOffset = kNumShadingFrames * gLatentDictionary.kNumLatentDims; // Offset to the first wo entry
            float wiEncoded[gLatentDictionary.kNumLatentDims];
            gLatentDictionary.getCodeBilinear(wiLocal[i], wiEncoded, wiInstance);
            [ForceUnroll]
            for (uint j = 0; j < gLatentDictionary.kNumLatentDims; j++)
            {
                input[gLatentDictionary.kNumLatentDims * i + j] = wiEncoded[j];
            }
            [ForceUnroll]
            for (uint j = 0; j < 3; j++)
            {
                input[woOffset + 3 * i + j] = woLocal[i][j];
            }
        }
    }
}

struct SphericalHarmonicsEncoding : IEncoding
{
    static const uint offset = kNumShadingFrames * 2 * gLatentSphericalHarmonics.kNumDims;
    [BackwardDifferentiable]
    static void prepareInput(
        float3 wiLocal[kNumShadingFrames],
        float3 woLocal[kNumShadingFrames],
        out float input[EvaluationDecoder.kNumInputs]
    )
    {
        [ForceUnroll]
        for (uint i = 0; i < kNumShadingFrames; i++)
        {
            uint i1 = i + 0;                 // Used to index the first half of the buffer
            uint i2 = i + kNumShadingFrames; // Used to index the second half of the buffer
            float wiEncoded[gLatentSphericalHarmonics.kNumDims];
            float woEncoded[gLatentSphericalHarmonics.kNumDims];
            gLatentSphericalHarmonics.getCode(wiLocal[i], wiEncoded);
            gLatentSphericalHarmonics.getCode(woLocal[i], woEncoded);
            [ForceUnroll]
            for (uint j = 0; j < gLatentSphericalHarmonics.kNumDims; j++)
            {
                input[gLatentSphericalHarmonics.kNumDims * i1 + j] = wiEncoded[j];
                input[gLatentSphericalHarmonics.kNumDims * i2 + j] = woEncoded[j];
            }
        }
    }
}

#if defined(USE_LATENT_DICTIONARY)
typedef LatentDictionaryEncoding Encoding;
#elif defined(USE_LATENT_BIPLANE)
typedef LatentBiplaneEncoding Encoding;
#elif defined(USE_SH_ENCODING)
typedef SphericalHarmonicsEncoding Encoding;
#else
typedef IdentityEncoding Encoding;
#endif

[BackwardDifferentiable]
float3 evaluationDecoder(no_diff float3 wi, no_diff float3 wo, float latentCode[LatentTexture.kNumLatentDims], uint batchIdx, uint warpIdx)
{
    Parameterization.transform(wi, wo);

    // Rotate (wi, wo) into the local shading frame(s).
    float3 wiLocal[kNumShadingFrames];
    float3 woLocal[kNumShadingFrames];
    ShadingFrame.toLocal(wi, wo, latentCode, batchIdx, warpIdx, wiLocal, woLocal);

    // Prepare the decoding network input vector by potentially applying some encoding on the preprocessed (wi, wo).
    float input[EvaluationDecoder.kNumInputs];
    Encoding.prepareInput(wiLocal, woLocal, input);

    [ForceUnroll]
    for (int i = 0; i < LatentTexture.kNumLatentDims; i++)
    {
        input[Encoding.offset + i] = latentCode[i];
    }

    // Run network.
    float output[3] = { 0.f, 0.f, 0.f };
    gEvaluationDecoder.eval(input, output, batchIdx, warpIdx);
    float3 out = float3(output[0], output[1], output[2]);

    // Final exponential activation.
    out = exp(out - 3.f);

#ifdef USE_COSINE_MODULATION
    out = out * woLocal[0].z;
#endif

    return out;
}

#define ACCUM_THREAD_GROUP_SIZE_X 32
#define ACCUM_THREAD_GROUP_SIZE_Y NUM_WARPS

[numthreads(ACCUM_THREAD_GROUP_SIZE_X, ACCUM_THREAD_GROUP_SIZE_Y, 1)]
void accumulateGradsPass(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint4 gridDim = gMaterialDataGenerator.gridDim;
    uint sampleCount = gMaterialDataGenerator.sampleCount;
    uint3 dispatchSize = uint3(gridDim.x * gridDim.z, gridDim.y * gridDim.w, sampleCount);
    if (any(dispatchThreadId >= dispatchSize))
        return;

    uint2 groupSize = uint2(ACCUM_THREAD_GROUP_SIZE_X, ACCUM_THREAD_GROUP_SIZE_Y);
    float invBatchSize = 1.f / (gridDim.x * gridDim.y * gridDim.z * gridDim.w * sampleCount);
    uint3 rngSeed = uint3(dispatchThreadId.xy, iteration);

    // Linear index over whole batch, used for storing training loss.
    uint batchIdx = dispatchSize.x * dispatchSize.y * dispatchThreadId.z + dispatchSize.x * dispatchThreadId.y + dispatchThreadId.x;

    // For TIN's gradient gradient reduction across warps.
    uint warpIdx = dispatchThreadId.y % NUM_WARPS;

    // Apply loss scale and perform reverse mode derivative propagation.
    float dOut[2] = { invBatchSize * evaluationLossScale, invBatchSize * samplingLossScale };
    __bwd_diff(evalModelAndLoss)(rngSeed, batchIdx, warpIdx, dispatchThreadId, groupSize, dOut);
}

void adamStep(inout float moment1, inout float moment2, inout float param, float derivative)
{
    // Adam optimizer step, see "Adam: A Method for Stochastic Optimization" by Kingma and Ba 2015.
    float t = iteration + 1.f;
    moment1 = gAdamParams.beta1 * moment1 + (1.f - gAdamParams.beta1) * derivative;
    moment2 = gAdamParams.beta2 * moment2 + (1.f - gAdamParams.beta2) * derivative * derivative;
    float mHat = moment1 / (1.f - pow(gAdamParams.beta1, t));
    float vHat = moment2 / (1.f - pow(gAdamParams.beta2, t));
    param = param - gAdamParams.learningRate * mHat / (sqrt(maxFloat(0.f, vHat)) + gAdamParams.epsilon);
}

void adamStep(inout float2 moment1, inout float2 moment2, inout float2 param, float2 derivative)
{
    // Adam optimizer step, see "Adam: A Method for Stochastic Optimization" by Kingma and Ba 2015.
    float t = iteration + 1.f;
    moment1 = gAdamParams.beta1 * moment1 + (1.f - gAdamParams.beta1) * derivative;
    moment2 = gAdamParams.beta2 * moment2 + (1.f - gAdamParams.beta2) * derivative * derivative;
    float2 mHat = moment1 / (1.f - pow(gAdamParams.beta1, t));
    float2 vHat = moment2 / (1.f - pow(gAdamParams.beta2, t));
    param = param - gAdamParams.learningRate * mHat / (sqrt(maxFloat2(float2(0.f), vHat)) + gAdamParams.epsilon);
}

[numthreads(16 * 16, 1, 1)]
void optimizeNetworksPass(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint numParamPairs[4] = {
        gEncoder.kNumParamPairs, gShadingFrameDecoder.kNumParamPairs, gEvaluationDecoder.kNumParamPairs, gSamplingDecoder.kNumParamPairs
    };
    uint paramPairsSum[4];
    paramPairsSum[0] = numParamPairs[0];
    paramPairsSum[1] = paramPairsSum[0] + numParamPairs[1];
    paramPairsSum[2] = paramPairsSum[1] + numParamPairs[2];
    paramPairsSum[3] = paramPairsSum[2] + numParamPairs[3];

    int paramId = int(dispatchThreadId.x);
    if (paramId >= paramPairsSum[3])
        return;

    // Fetch parameter(s) and the corresponding derivative(s) and moments.
    float2 param, deriv;
    float4 moments;
    if (paramId < paramPairsSum[0])
        gEncoder.getOptimizerState(paramId, param, deriv, moments);
    else if (paramId < paramPairsSum[1])
        gShadingFrameDecoder.getOptimizerState(paramId - paramPairsSum[0], param, deriv, moments);
    else if (paramId < paramPairsSum[2])
        gEvaluationDecoder.getOptimizerState(paramId - paramPairsSum[1], param, deriv, moments);
    else if (paramId < paramPairsSum[3])
        gSamplingDecoder.getOptimizerState(paramId - paramPairsSum[2], param, deriv, moments);

    // Cancel out loss scale again.
    float lossScale = paramId < paramPairsSum[2] ? evaluationLossScale : samplingLossScale;
    deriv /= lossScale;

    // Optionally skip invalid Inf/NaN values that pop up.
#ifdef SKIP_INVALID_GRADIENTS
    if (any(!isfinite(deriv)))
        deriv = 0.f;
#endif

    // Potentially apply L2 regularization of the parameters, by directly modifying the derivatives.
    deriv += l2ParameterRegularization * param;

    // Run the actual adam update step, working on two parameters at a time.
    adamStep(moments.xz, moments.yw, param, deriv);

    // Write back updated parameter(s) and moments.
    if (paramId < paramPairsSum[0])
        gEncoder.updateOptimizerState(paramId, param, moments);
    else if (paramId < paramPairsSum[1])
        gShadingFrameDecoder.updateOptimizerState(paramId - paramPairsSum[0], param, moments);
    else if (paramId < paramPairsSum[2])
        gEvaluationDecoder.updateOptimizerState(paramId - paramPairsSum[1], param, moments);
    else if (paramId < paramPairsSum[3])
        gSamplingDecoder.updateOptimizerState(paramId - paramPairsSum[2], param, moments);
}

[numthreads(256, 1, 1)]
void optimizeLatentTexturePass(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint width = LatentTexture.kWidth >> activeLevel;
    const uint height = LatentTexture.kHeight >> activeLevel;
    const uint index = dispatchThreadId.x;
    if (index >= width * height)
        return;
    uint2 texel = uint2(index % width, index / width);

    float param, deriv;
    float2 moments;
    for (int ch = 0; ch < LatentTexture.kNumLatentDims; ch++)
    {
        gLatentTexture.getOptimizerState(activeTile, activeLevel, ch, texel, param, deriv, moments);
        deriv /= evaluationLossScale; // Cancel out loss scale again.
#ifdef SKIP_INVALID_GRADIENTS
        if (!isfinite(deriv))
            deriv = 0.f;
#endif
        if (deriv == 0.f)
            continue; // This isn't done in Pytorch, but saves a lot of work as it skips updating the many zero parameters.
        adamStep(moments.x, moments.y, param, deriv);
        gLatentTexture.updateOptimizerState(activeTile, activeLevel, ch, texel, param, moments);
    }
}

[numthreads(16, 16, 1)]
void optimizeLatentHashGridPass(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 index = dispatchThreadId.xy;
    if (any(index >= gLatentHashGrid.getBufferSize(activeLevel)))
        return;

    float param, deriv;
    float2 moments;
    for (int ch = 0; ch < LatentHashGrid.kNumLatentDimsPerLevel; ch++)
    {
        gLatentHashGrid.getOptimizerState(activeLevel, ch, index, param, deriv, moments);
        deriv /= evaluationLossScale; // Cancel out loss scale again.
#ifdef SKIP_INVALID_GRADIENTS
        if (!isfinite(deriv))
            deriv = 0.f;
#endif
        if (deriv == 0.f)
            continue; // This isn't done in Pytorch, but saves a lot of work as it skips updating the many zero parameters.
        adamStep(moments.x, moments.y, param, deriv);
        gLatentHashGrid.updateOptimizerState(activeLevel, ch, index, param, moments);
        float test[1];
    }
}

[numthreads(16, 16, 1)]
void optimizeLatentDictionaryPass(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 index = dispatchThreadId.xy;
    if (any(index >= gLatentDictionary.getSize(activeLevel)))
        return;

    float param, deriv;
    float2 moments;
    for (int ch = 0; ch < gLatentDictionary.kNumLatentDimsPerLevel; ch++)
    {
        gLatentDictionary.getOptimizerState(instance, activeLevel, ch, index, param, deriv, moments);
        deriv /= evaluationLossScale; // Cancel out loss scale again.
#ifdef SKIP_INVALID_GRADIENTS
        if (!isfinite(deriv))
            deriv = 0.f;
#endif
        if (deriv == 0.f)
            continue; // This isn't done in Pytorch, but saves a lot of work as it skips updating the many zero parameters.
        adamStep(moments.x, moments.y, param, deriv);
        gLatentDictionary.updateOptimizerState(instance, activeLevel, ch, index, param, moments);
    }
}

[numthreads(256, 1, 1)]
void evalModelPass(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    if (idx >= evalBufferSize)
        return;

    float2 uv = uvBuffer[idx];
    float3 wi = wiBuffer[idx];
    float3 wo = woBuffer[idx];

    // Evaluate the reference model.
    SampleBufferContentDesc fixedInputs = SampleBufferContentDesc(sampleBufferContent);
    MaterialSample ms = gMaterialDataGenerator.generateMaterialSample(dispatchThreadId, uint2(16 * 16, 1), fixedInputs, uv, wi, wo);
    targetEvalBuffer[idx] = ms.weight;

    // Depending on the `sampleBufferContent`, this might have sampled new uv/wi/wo.
    uv = ms.uv;
    wi = ms.wi;
    wo = ms.wo;

    // Retrieve the target normal(s) for visualization.
    for (uint i = 0; i < kNumExportedBsdfs; i++)
    {
        targetNormalBuffer[idx * kNumExportedBsdfs + i] = ms.normal[i];
    }

    // Fetch latent code for the specified MIP level.
    float latentCode[LatentTexture.kNumLatentDims];
    if (fetchLatentsFromTexture == 1)
    {
        // Latents are available and can just be fetched. E.g. for plotting during checkpoints.
        gLatentTexture.getCodeBilinear(uv, activeLevel, latentCode);
    }
    else
    {
        // Latents might have to be encoded first. E.g. when computing validation losses outside of checkpoints.
        fetchLatentCode(ms, 0.f, 0, 0, latentCode);
    }

    // Evaluate the learned model.
    learnedEvalBuffer[idx] = evaluationDecoder(wi, wo, latentCode, idx, 0);

    // Evaluate the learned sampling density.
    var params = LearnedSampler.decodeParams(wi, latentCode, idx, 0);
    learnedPdfBuffer[idx] = LearnedSampler.pdf(wi, wo, params);
}

#define LATENT_GEN_THREAD_GROUP_SIZE 16

[numthreads(LATENT_GEN_THREAD_GROUP_SIZE, LATENT_GEN_THREAD_GROUP_SIZE, 1)]
void generateLatentTexturePass(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint width = LatentTexture.kWidth >> activeLevel;
    const uint height = LatentTexture.kHeight >> activeLevel;
    const uint2 texel = dispatchThreadId.xy;
    if (texel.x >= width || texel.y >= height)
        return;

    // Generate MaterialSample based on threadId.
    float2 uvFixed;
    float3 wiFixed;
    float3 woFixed;
    SampleBufferContentDesc fixedInputs = SampleBufferContentDesc::None;
    MaterialSample ms = gMaterialDataGenerator.generateMaterialSample(
        dispatchThreadId, uint2(LATENT_GEN_THREAD_GROUP_SIZE), fixedInputs, uvFixed, wiFixed, woFixed
    );

    // Generate the latent code.
    float latentCode[LatentTexture.kNumLatentDims];
#if defined(USE_LATENT_HASH_GRID)
    gLatentHashGrid.getCodeBilinear(ms.uv, latentCode);
#elif defined(USE_LATENT_ENCODER)
    float encInput[Encoder.kNumInputs];
    createEncoderInput(ms, encInput);
    gEncoder.eval(encInput, latentCode, 0, 0);
#endif

    for (int ch = 0; ch < LatentTexture.kNumLatentDims; ch++)
    {
        gLatentTexture.setTexel(activeTile, activeLevel, ch, texel, latentCode[ch]);
    }
}
