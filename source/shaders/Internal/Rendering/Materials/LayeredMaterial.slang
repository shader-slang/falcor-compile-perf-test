/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
import Internal.MaterialNetwork.MaterialX.MxCommon.MxLayeredMaterialInstance;
import Internal.MaterialNetwork.MaterialX.MxCommon.MxWeights;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;
import Scene.Scene;
import Rendering.Materials.BSDFs.LambertDiffuseBRDF;
import Rendering.Materials.BSDFs.LambertDiffuseBTDF;
import Rendering.Materials.PBRT.PBRTDielectricMaterialInstance;
import Rendering.Materials.BSDFs.SpecularMicrofacet;
import Rendering.Materials.BSDFs.SimpleBTDF;
import Rendering.Materials.BSDFs.OrenNayarBRDF;
import Rendering.Materials.PBRT.PBRTConductorMaterialInstance;

/**
 * This is an example of the per-material layered weight calculator.
 * The layered program comes from the LayeringApp.exe tool and needs to be manually copied here.
 * This is intended purely for debugging and research, the main workflow should go through MaterialX,
 * where these calculators are generated automatically.
 */
struct ExampleLayeredWeightCalculator : IMxLayeredWeightCalculator
{
    // clang-format off
    // These functions need to be filled in with the output of LayeringApp.exe
    void calculateWeights<let TBsdfCount : int, let TLayerCount : int>( const MxLayeredMaterialData<TBsdfCount, TLayerCount> data, float3 wiLocal[TBsdfCount], bool isFrontFacing, out MxWeights<TBsdfCount> weights )
    {
        if (isFrontFacing)
        {
            // This is the forward program
            const float3 Wlb0 = data.bsdfWeights[0];
            const float3 Wlb1 = data.bsdfWeights[1];
            const float3 Wly0 = data.layerWeights[0];


            const float3 n0 = Wlb0;
            const float3 n1 = Wlb1;
            const float3 n2 = Wly0;
            const float3 n3 = (n0 + n1);
            const float3 n4 = (n0 / n3);
            const float3 n5 = (n1 / n3);
            const float3 n6 = (n2 * n4);
            const float3 n7 = (n2 * n5);
            weights.bsdfWeights[0] = n6;
            weights.bsdfWeights[1] = n7;
        }
        else
        {
            // This is the backward program
            const float3 Wlb0 = data.bsdfWeights[0];
            const float3 Wlb1 = data.bsdfWeights[1];
            const float3 Wly0 = data.layerWeights[0];


            const float3 n0 = Wlb0;
            const float3 n1 = Wlb1;
            const float3 n2 = Wly0;
            const float3 n3 = (n0 + n1);
            const float3 n4 = (n0 / n3);
            const float3 n5 = (n1 / n3);
            const float3 n6 = (n2 * n4);
            const float3 n7 = (n2 * n5);
            weights.bsdfWeights[0] = n6;
            weights.bsdfWeights[1] = n7;
        }
    }

    void calculatePlainWeights<let TBsdfCount : int, let TLayerCount : int>( const MxLayeredMaterialData<TBsdfCount, TLayerCount> data, out MxWeights<TBsdfCount> weights )
    {
        // This is the PureWeight program
        const float3 Wlb0 = data.bsdfWeights[0];
        const float3 Wlb1 = data.bsdfWeights[1];
        const float3 Wly0 = data.layerWeights[0];


        const float3 n0 = Wlb0;
        const float3 n1 = Wlb1;
        const float3 n2 = Wly0;
        const float3 n3 = (n0 + n1);
        const float3 n4 = (n0 / n3);
        const float3 n5 = (n1 / n3);
        const float3 n6 = (n2 * n4);
        const float3 n7 = (n2 * n5);
        weights.bsdfWeights[0] = n6;
        weights.bsdfWeights[1] = n7;
    }
    // clang-format on
}

/**
 * This is a hard-coded layered material which in the future will be auto-generated. To generate your own layered
 * material, the layer tree needs to be created, and the code which is generated needs to be copied into a
 * IMxLayeredWeightCalculator struct and the inputs need to be adjusted. This is only for debugging and performance
 * analysis; once we are sure that the rest of the system works, this could be removed
 */
struct LayeredMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    typealias LayeredMaterialInstance = MxLayeredMaterialInstance<2, 1, ExampleLayeredWeightCalculator>;
    typealias LayeredMaterialData = MxLayeredMaterialData<2, 1>;
    typedef LayeredMaterialInstance MaterialInstance;

    LayeredMaterialInstance setupMaterialInstance(
        const MaterialSystem ms,
        const ShadingData sd,
        const ITextureSampler lod,
        const uint hints
    )
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        LambertDiffuseBRDF bsdf0 = { float3(1.0f, 0.0f, 0.0f) };
        LambertDiffuseBRDF bsdf1 = { float3(0.0f, 0.0f, 1.0f) };

        IBSDF bsdfs[2];
        bsdfs = { bsdf0, bsdf1 };

        // Input data
        LayeredMaterialData d;
        d.bsdfWeights = { 1.0f, 1.0f };
        d.layerWeights = { 1.0f };

        d.bsdfs = bsdfs;
        d.frontFacing = sd.frontFacing;
        d.transmits = { 0.0, 0.0 };

        // create BSDFs - these should be autoloaded
        LayeredMaterialInstance mi;
        mi.data = d;

        return mi;
    }
};
