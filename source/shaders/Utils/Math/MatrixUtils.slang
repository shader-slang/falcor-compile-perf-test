/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
float3x3 float3x3FromCols(float3 col0, float3 col1, float3 col2)
{
    float3x3 result;
    result = transpose(float3x3(col0, col1, col2));
    return result;
}

float3 extractTranslate(float4x4 m)
{
    return float3(m[0].w, m[1].w, m[2].w);
}

// Transform that goes from space Y to space X,
// with vectors going on the right A_inX = xform * A_inY;
struct xform4x4
{
    row_major float4x4 xform;

    __init(float4x4 m) { xform = m; }
}

// Affine transform
struct xform3x4
{
    row_major float3x4 xform;

    __init(float3x4 m) { xform = m; }

    __init(float4x4 m) { xform = float3x4(m[0], m[1], m[2]); }
}

xform3x4 concat(xform3x4 aFromB, xform3x4 bFromC)
{
    xform3x4 aFromC;
    aFromC.xform = (float3x4)mul(
        float4x4(aFromB.xform[0], aFromB.xform[1], aFromB.xform[2], float4(0, 0, 0, 1)),
        float4x4(bFromC.xform[0], bFromC.xform[1], bFromC.xform[2], float4(0, 0, 0, 1))
    );
    return aFromC;
}

float3 transformPoint(xform4x4 aFromB, float3 pointB)
{
    float4 hpoint = mul(aFromB.xform, float4(pointB, 1.f));
    return hpoint.xyz / hpoint.w;
}

float3 transformPointAffine(xform4x4 aFromB, float3 pointB)
{
    float4 hpoint = mul(aFromB.xform, float4(pointB, 1.f));
    return hpoint.xyz;
}

float3 transformPoint(xform3x4 aFromB, float3 pointB)
{
    return mul(aFromB.xform, float4(pointB, 1.f)).xyz;
}

float3 transformVector(xform4x4 aFromB, float3 vectorB)
{
    return mul(aFromB.xform, float4(vectorB, 0.f)).xyz;
}

float3 transformVector(xform3x4 aFromB, float3 vectorB)
{
    return mul(aFromB.xform, float4(vectorB, 0.f)).xyz;
}

__generic<T : __BuiltinFloatingPointType, let N : int> extension matrix<T, N, N>
{
    // Create an identity matrix of size N and type T
    static matrix<T, N, N> identity()
    {
        static matrix<T, N, N> m = matrix<T, N, N>((T)0);
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            m[i][i] = (T)1;
        return m;
    }
}
