/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Unit tests for RMW operations on different buffer types.

    The updateBuffer() kernel is run multiple times followed by readBuffer()
    that copies the result to the output buffer.
*/

#if TYPE == 0
RWByteAddressBuffer buffer;
#elif TYPE == 1
RWBuffer<uint> buffer;
#elif TYPE == 2
RWStructuredBuffer<uint> buffer;
#endif

RWStructuredBuffer<uint> result;

// Clear kernel.
// This is necessary for RWStructuredBuffer as we can't use ID3D12CommandList::ClearUnorderedAccessViewUint for structured buffers.
[numthreads(256, 1, 1)]
void clearBuffer(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
#if TYPE == 0
    buffer.Store(i * 4, 0);
#else
    buffer[i] = 0;
#endif
}

[numthreads(256, 1, 1)]
void updateBuffer(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
#if TYPE == 0
    uint val = buffer.Load(i * 4);
    val += i + 1;
    buffer.Store(i * 4, val);
#else
    buffer[i] += i + 1;
#endif
}

[numthreads(256, 1, 1)]
void readBuffer(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
#if TYPE == 0
    result[i] = buffer.Load(i * 4);
#else
    result[i] = buffer[i];
#endif
}
