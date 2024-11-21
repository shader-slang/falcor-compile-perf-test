/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
struct VsOut
{
    float2 texC : TEXCOORD;
    float4 posH : POSITION;
};

struct GsOut
{
    float2 texC : TEXCOORD;
    float4 posH : SV_POSITION;
    uint rtIndex : SV_RenderTargetArrayIndex;
};

[maxvertexcount(_OUTPUT_VERTEX_COUNT)]
void main(triangle VsOut input[3], inout TriangleStream<GsOut> outStream)
{
    GsOut output;
    uint mask = _VIEWPORT_MASK;

    while (mask != 0)
    {
        uint layer = firstbitlow(mask);

        for (int i = 0; i < 3; i++)
        {
            output.rtIndex = layer;
            output.posH = input[i].posH;
            output.texC = input[i].texC;
            outStream.Append(output);
        }
        outStream.RestartStrip();
        mask &= ~(1 << layer);
    }
}