/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#ifndef LATENT_DIMS
#error LATENT_DIMS not defined
#endif

struct NeuralMaterialUpdater
{
    uint2 dims;
    uint mipLevel;

    Texture2D<float> channels[LATENT_DIMS];
    RWTexture2D<float4> latentTextures[LATENT_DIMS / 4];

    void execute(const uint2 pixel)
    {
        if (any(pixel >= dims))
            return;

        [ForceUnroll]
        for (uint i = 0; i < LATENT_DIMS / 4; i++)
        {
            float4 c = {
                channels[i * 4 + 0][pixel],
                channels[i * 4 + 1][pixel],
                channels[i * 4 + 2][pixel],
                channels[i * 4 + 3][pixel],
            };
            latentTextures[i][pixel] = c;
        }
    }
};

ParameterBlock<NeuralMaterialUpdater> gUpdater;

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    gUpdater.execute(threadID.xy);
}
