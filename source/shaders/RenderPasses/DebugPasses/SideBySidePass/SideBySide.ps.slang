/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import "RenderPasses/DebugPasses/Comparison.ps";

struct SideBySide : ICalcPixelColor
{
    float2x4 calcColors(uint2 pixelPos)
    {
        float2x4 colors;
        colors[0] = gLeftInput[uint2(pixelPos.x + gLeftBound, pixelPos.y)];
        colors[1] = gRightInput[uint2(int(pixelPos.x) - gSplitLocation + int(gLeftBound), pixelPos.y)];
        return colors;
    }
};

float4 main(float2 texC: TEXCOORD, float4 pos: SV_Position) : SV_Target0
{
    SideBySide comp = {};
    return compare(texC, pos, comp);
}
