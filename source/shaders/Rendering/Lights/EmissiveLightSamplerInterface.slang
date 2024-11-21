/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;

/** Slang interface and structs used by emissive light samplers.
*/

/** Describes a light sample on an emissive triangle.
    If pdf == 0.0 is returned, the sample in invalid and the other fields should not be used.
*/
struct TriangleLightSample
{
    uint    triangleIndex;      ///< Index of the sampled triangle.
    float3  posW;               ///< Sampled point on the light source in world space.
    float3  normalW;            ///< Normal of the sampled point on the light source in world space.
    float3  dir;                ///< Normalized direction from the shading point to the sampled point on the light source in world space.
    float   distance;           ///< Distance from the shading point to the sampled point.
    float3  Le;                 ///< Emitted radiance. This is zero if the light is back-facing or sample is invalid.
    float   pdf;                ///< Probability density with respect to solid angle from the shading point. The range is [0,inf] (inclusive), where pdf == 0.0 indicated an invalid sample.
    float2  uv;                 ///< Light sample barycentric coords over the triangle
};

/** Describes a light sample at a hit point on an emissive triangle.
    This is used for PDF evaluation.
*/
struct TriangleLightHit
{
    uint    triangleIndex;      ///< Index of the triangle.
    float3  posW;               ///< Sampled point on the triangle in world space.
    float3  normalW;            ///< Face normal of the triangle in world space, flipped for back-facing hits.
};

/** Slang interface for emissive light sampler implementations.
*/
interface IEmissiveLightSampler
{
    /** Draw a single light sample.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Light sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleLight<S : ISampleGenerator>(const float3 posW, const float3 normalW, const bool upperHemisphere, inout S sg, out TriangleLightSample ls);

    /** Evaluate the PDF associated with selecting an emissive triangle.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in] triangleIndex index of selected triangle
        \return Probability of selecting the input triangle
    */
    float evalTriangleSelectionPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const uint triangleIndex);

    /** Evaluate the PDF at a shading point given a hit point on an emissive triangle.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in] hit Triangle hit data.
        \return Probability density with respect to solid angle at the shading point.
    */
    float evalPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const TriangleLightHit hit);
};
