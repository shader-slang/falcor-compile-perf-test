/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

Buffer<float> gIesData;
RWTexture2D<float> gTexture;
RWTexture2D<float> gFluxTexture;

cbuffer CB
{
    uint gBakeResolution;
};

float findAngleIndex(float angle, int offset, int count)
{
    if (count == 1) return 0;

    float left;
    float right = gIesData[offset];

    if (angle <= right) return 0;

    for(int i = 1; i < count; i++)
    {
        left = right;
        right = gIesData[offset + i];

        if (angle >= left && angle <= right)
        {
            return float(i - 1) + ((right > left) ? (angle - left) / (right - left) : 0.0);
        }
    }

    return count - 1;
}

[numthreads(16, 16, 1)]
void main(uint2 threadID : SV_DispatchThreadID)
{
    int numVerticalAngles = int(gIesData[3]);
    int numHorizontalAngles = int(gIesData[4]);
    int headerSize = 13;

    float verticalAngle = float(threadID.x) * (180.f / float(gBakeResolution));
    float horizontalAngle = float(threadID.y) * (360.f / float(gBakeResolution)) - 180.f;

    float lastVerticalAngle = gIesData[headerSize + numVerticalAngles - 1];
    float lastHorizontalAngle = gIesData[headerSize + numVerticalAngles + numHorizontalAngles - 1];

    if (verticalAngle > lastVerticalAngle)
    {
        gTexture[threadID] = 0;
        return;
    }

    if (lastHorizontalAngle <= 180.f)
    {
        // Apply symmertry
        horizontalAngle = abs(horizontalAngle);
        if (lastHorizontalAngle == 90.f && horizontalAngle > 90.f)
        {
            horizontalAngle = 180.f - horizontalAngle;
        }
    }
    else
    {
        // No symmetry, but the profile has data in 0..360 degree range, convert our -180..180 range to that
        if (horizontalAngle < 0) horizontalAngle += 360.f;
    }

    float verticalAngleIndex = findAngleIndex(verticalAngle, headerSize, numVerticalAngles);
    float horizontalAngleIndex = findAngleIndex(horizontalAngle, headerSize + numVerticalAngles, numHorizontalAngles);

    int dataOffset = headerSize + numHorizontalAngles + numVerticalAngles;

    float a = gIesData[dataOffset + int(floor(horizontalAngleIndex)) * numVerticalAngles + int(floor(verticalAngleIndex))];
    float b = gIesData[dataOffset + int(floor(horizontalAngleIndex)) * numVerticalAngles + int(ceil(verticalAngleIndex))];
    float c = gIesData[dataOffset + int(ceil(horizontalAngleIndex)) * numVerticalAngles + int(floor(verticalAngleIndex))];
    float d = gIesData[dataOffset + int(ceil(horizontalAngleIndex)) * numVerticalAngles + int(ceil(verticalAngleIndex))];

    float candelas = lerp(
        lerp(a, b, frac(verticalAngleIndex)),
        lerp(c, d, frac(verticalAngleIndex)),
        frac(horizontalAngleIndex));

    float normalization = gIesData[0];

    float result = candelas * normalization;

    gTexture[threadID] = result;

    // Compute the flux factor for this profile.
    // The flux factor is the integral of the profile over the directions of the sphere.
    // The values stored in gFluxTexture are scaled such that the sum of all pixels in the texture equals the integral.
    float theta = verticalAngle / 180.f * M_PI;
    gFluxTexture[threadID] = result * sin(theta) * 2.f * M_PI * M_PI / (gBakeResolution * gBakeResolution);
}
