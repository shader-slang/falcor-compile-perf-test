/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import PathVertex;

cbuffer CB
{
    struct Training {
        StructuredBuffer<float3>              radiance;       ///< Outgoing radiance from a path vertex
        StructuredBuffer<RadianceParams>      radianceParams;   ///< Path vertex properties for conditioning the neural net queries.
        Buffer<uint>                          recordCount;
    } gTraining;
    struct Shuffled {
        RWStructuredBuffer<float3>            radiance;       ///< Outgoing radiance from a path vertex
        RWStructuredBuffer<RadianceParams>    radianceParams;   ///< Path vertex properties for conditioning the neural net queries.
    } gShuffled;
};

[numthreads(SHUFFLE_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint src = dispatchThreadId.x;
    uint recordCount = gTraining.recordCount[0];
    if (src >= recordCount) return;

    uint prime = 2000000011;
    uint large = 1000003;
    uint dst = (prime * src + large) % recordCount;
    gShuffled.radianceParams[dst] = gTraining.radianceParams[src];
    gShuffled.radiance[dst] = gTraining.radiance[src];
}
