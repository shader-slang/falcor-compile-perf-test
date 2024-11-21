/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
cbuffer GlobalCB
{
    int gSplitLocation;   ///< X-value of the current split location to display
    uint gDividerSize;    ///< How wide should the divider be?
    float4 gDividerColor; ///< What color should the divider be this frame?
    int2 gMousePosition;  ///< What is the current position of the mouse? (Used to position arrows)
    bool gDrawArrows;     ///< Should we draw arrows to the left & right of the divider?
    uint gLeftBound;      ///< How many pixels from the left side should the comparison window start?
};

Texture2D gLeftInput;
Texture2D gRightInput;
Texture2D<float> gArrowTex;

interface ICalcPixelColor
{
    float2x4 calcColors(uint2 pixelPos);
};

float4 compare<C : ICalcPixelColor>(float2 texC, float4 pos, C calc)
{
    // Get the two sides of the image
    uint2 pixelPos = (uint2)pos.xy;
    float2x4 colors = calc.calcColors(pixelPos);
    float4 leftColor = colors[0];
    float4 rightColor = colors[1];

    // Combine two images, depending on which side of the split we're on.  Clamp inputs to minimum 0.
    float4 color = max((pixelPos.x < gSplitLocation) ? leftColor : rightColor, float4(0, 0, 0, 0));

    // Overlay divider onto the image
    if (abs(int(pixelPos.x) - gSplitLocation) < gDividerSize)
    {
        color = gDividerColor;
    }

    // Draw arrows
    if (gDrawArrows)
    {
        uint2 dims;
        gArrowTex.GetDimensions(dims.x, dims.y);
        int2 arrow1Coord = int2(pixelPos.x - 5 - gSplitLocation, pixelPos.y - gMousePosition.y + 8);
        if (all(clamp(arrow1Coord, int2(0, 0), int2(dims)) == arrow1Coord))
        {
            float arrowAlpha = gArrowTex[arrow1Coord];
            color = gDividerColor * arrowAlpha + ((arrowAlpha < 1.0) ? color * (1 - arrowAlpha) : float4(0, 0, 0, 0));
        }

        int2 arrow2Coord = int2(pixelPos.x + 21 - gSplitLocation, pixelPos.y - gMousePosition.y + 8);
        if (all(clamp(arrow2Coord, int2(0, 0), int2(dims)) == arrow2Coord))
        {
            arrow2Coord.x = abs(arrow2Coord.x - 15);
            float arrowAlpha = gArrowTex[arrow2Coord];
            color = gDividerColor * arrowAlpha + ((arrowAlpha < 1.0) ? color * (1 - arrowAlpha) : float4(0, 0, 0, 0));
        }
    }

    return color;
}
