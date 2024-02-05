/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Rendering.Materials.IBSDF;
import Utils.SlangUtils;

/// Per-bsdfs weights, result of the layering programs.
struct MxWeights<let TBsdfCount : int>
{
    float3 bsdfWeights[TBsdfCount];
}

struct MxSubsurfaceData
{
    float3 color;
    float3 radius;
    float3 N;
    float3 T;
    float3 w0;
    float3 meancosine0;
    float3 meancosine1;

    /// This allows using the same user input string to setup `DualHenyeyGreensteinPhaseFunction` and this class
    [[mutating]]
    void dual_hg(float3 meancosine0_, float3 meancosine1_, float3 w0_)
    {
        meancosine0 = meancosine0_;
        meancosine1 = meancosine1_;
        w0 = w0_;
    }

    [[mutating]]
    void single_hg(float meancosine0_)
    {
        meancosine0 = meancosine0_;
        meancosine1 = meancosine0_;
        w0 = 1.f;
    }
}

interface IMxLayeredMaterialData
{
    int getBsdfCount();
    int getLayerCount();
    IBSDF getBsdf(int bsdfIndex);

    float3 getBsdfWeight(int bsdfIndex);
    float3 getLayerWeight(int layerIndex);
    float3 getBsdfN(int bsdfIndex);
    float3 getBsdfT(int bsdfIndex);
    float getTransmits(int bsdfIndex);
    bool getIsFrontFacing();

    bool hasSubsurfaceData();
    MxSubsurfaceData getSubsurfaceData();

    // These are from IBSDF, just doing static dispatch here
    float3 eval<S : ISampleGenerator>(int bsdfIndex, const float3 wi, const float3 wo, inout S sg);
    bool sample<S : ISampleGenerator>(
        int bsdfIndex,
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out uint lobeType,
        inout S sg
    );
    float evalPdf(int bsdfIndex, const float3 wi, const float3 wo);
    AlbedoContributions evalAlbedo(int bsdfIndex, const float3 wi, const LobeType lobetype);
    RoughnessInformation getRoughnessInformation(int bsdfIndex, const float3 wi);
}

/// Output of the shader graph, input of the layering programs.
struct MxLayeredMaterialData<let TBsdfCount : int, let TLayerCount : int> : IMxLayeredMaterialData
{
    float3 bsdfWeights[TBsdfCount];                      ///< Weights of each BSDFs pre layering
    float3 layerWeights[ArrayMax<1, TLayerCount>.value]; ///< Weight of each layer
    static const int bsdfCount = TBsdfCount;             ///< Number of BSFDs
    static const int layerCount = TLayerCount;           ///< Number of layers
    IBSDF bsdfs[TBsdfCount];                             ///< The individual BSDFs produced by the shaders.
    float3 bsdfN[TBsdfCount];                            ///< Per-lobe normals, defaults to sd.frame.N (always valid and normalized)
    float3 bsdfT[TBsdfCount];                            ///< Per-lobe tangents, defaults to sd.frame.T (always valid and normalized)
    float transmits[TBsdfCount];                         ///< if flag is true, the BTDF will contribute to transmission (if possible)
    bool frontFacing;                                    ///< which side of the layer stack are we evaluating?

    [mutating]
    void setBsdf(int index, IBSDF bsdf) { bsdfs[index] = bsdf; }

    int getBsdfCount() { return bsdfCount; }
    int getLayerCount() { return layerCount; }
    IBSDF getBsdf(int bsdfIndex) { return bsdfs[bsdfIndex]; }

    float3 getBsdfWeight(int bsdfIndex) { return bsdfWeights[bsdfIndex]; }
    float3 getLayerWeight(int layerIndex) { return layerWeights[layerIndex]; }
    float3 getBsdfN(int bsdfIndex) { return bsdfN[bsdfIndex]; }
    float3 getBsdfT(int bsdfIndex) { return bsdfT[bsdfIndex]; }
    float getTransmits(int bsdfIndex) { return transmits[bsdfIndex]; }
    bool getIsFrontFacing() { return frontFacing; }

    bool hasSubsurfaceData() { return false; }
    MxSubsurfaceData getSubsurfaceData() { return {}; }

    // These are from IBSDF, just doing static dispatch here
    float3 eval<S : ISampleGenerator>(int bsdfIndex, const float3 wi, const float3 wo, inout S sg)
    {
        return bsdfs[bsdfIndex].eval(wi, wo, sg);
    }
    bool sample<S : ISampleGenerator>(
        int bsdfIndex,
        const float3 wi,
        out float3 wo,
        out float pdf,
        out float3 weight,
        out uint lobeType,
        inout S sg
    )
    {
        return bsdfs[bsdfIndex].sample(wi, wo, pdf, weight, lobeType, sg);
    }
    float evalPdf(int bsdfIndex, const float3 wi, const float3 wo) { return bsdfs[bsdfIndex].evalPdf(wi, wo); }
    AlbedoContributions evalAlbedo(int bsdfIndex, const float3 wi, const LobeType lobetype)
    {
        return bsdfs[bsdfIndex].evalAlbedo(wi, lobetype);
    }
    RoughnessInformation getRoughnessInformation(int bsdfIndex, const float3 wi) { return bsdfs[bsdfIndex].getRoughnessInformation(wi); }
}

/**
 * Interface to calculate weights. This is not intended to be dynamic dispatched ever.
 * There will be [specialize] keyword in the future to turn any dynamic dispatch into an error.
 * This is used inside MxLayeredMaterialInstance.
 * The actual implementation depends on the shader graph and is generated as part of the MaterialX codegen.
 * (The actual code will be provided by Andrea's layering branch)
 */
interface IMxLayeredWeightCalculator
{
    void calculateWeights<let TBsdfCount : int, let TLayerCount : int, TLayeredMaterialData : IMxLayeredMaterialData>(
        const TLayeredMaterialData data,
        float3 wiLocal[TBsdfCount],
        bool isFrontFacing,
        out MxWeights<TBsdfCount> weights
    );
    void calculatePlainWeights<let TBsdfCount : int, let TLayerCount : int, TLayeredMaterialData : IMxLayeredMaterialData>(
        const TLayeredMaterialData data,
        out MxWeights<TBsdfCount> weights
    );
}
