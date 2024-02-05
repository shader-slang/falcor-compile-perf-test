/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "DepthOfFieldDefines.slangh"

RWByteAddressBuffer gRWExecuteIndirectBuffer;

[numthreads(32, 1, 1)]
[shader("compute")]
void resetIndirect(uint globalId: SV_DispatchThreadID)
{
    if (globalId < BOKEH_MIP_LEVELS + 1)
    {
        // Reset indirect draw arguments based on D3D12_DRAW_INDEXED_ARGUMENTS struct layout.
        gRWExecuteIndirectBuffer.Store(INDIRECT_SEPARATION * globalId, 6);      // IndexCountPerInstance
        gRWExecuteIndirectBuffer.Store4(INDIRECT_SEPARATION * globalId + 4, 0); // InstanceCount (this is the atomic counter),
                                                                                // StartIndexLocation, BaseVertexLocation,
                                                                                // StartInstanceLocation
    }
}
