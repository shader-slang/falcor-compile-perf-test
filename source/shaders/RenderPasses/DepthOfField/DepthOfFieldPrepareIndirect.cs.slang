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

/** Shader for preparing arguments for indirect dispatch of compute splat pass.
    This computes the number of thread groups to dispatch based on the splat counts.
*/
[numthreads(1, 1, 1)]
[shader("compute")]
void prepareIndirect(uint globalId: SV_DispatchThreadID)
{
    if (globalId < 1)
    {
        uint splatCount = gRWExecuteIndirectBuffer.Load(INDIRECT_SEPARATION * BOKEH_MIP_LEVELS + 4);

        // Compute number of thread groups. The thread group size of the splat pass is (256, 1, 1).
        uint threadGroupCount = (splatCount + 255) / 256;
        threadGroupCount = min(threadGroupCount, 65535); // This is a D3D12 limit. We don't need that many thread groups, but just in case
                                                         // someone tries.

        // Placing DispatchArguments in the second half of the buffer:
        // ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ
        gRWExecuteIndirectBuffer.Store3(INDIRECT_SEPARATION * (BOKEH_MIP_LEVELS + 1), uint3(threadGroupCount, 1, 1));
    }
}
