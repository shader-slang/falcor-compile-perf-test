/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.LobeType;
__exported import Utils.Sampling.SampleGeneratorInterface;

/// Minimum cos(theta) for the incident and outgoing vectors.
/// Some BSDF functions are not robust for cos(theta) == 0.0,
/// so using a small epsilon for consistency.
/// TODO: Move into IBSDF if possible
static const float kMinCosTheta = 1e-6f;

/** Low-level interface for BSDF functions.

    Conventions:
    - All operations are done in a local coordinate frame.
    - The local frame has normal N=(0,0,1), tangent T=(1,0,0) and bitangent B=(0,1,0).
    - The incident and outgoing direction point away from the shading location.
    - The incident direction (wi) is always in the positive hemisphere.
    - The outgoing direction (wo) is sampled.
    - Evaluating the BSDF always includes the foreshortening term (dot(wo, n) = wo.z).
*/

/** Albedo is split into four components. They have to sum up to 1.0.
*/
struct AlbedoContributions
{
    float3  reflection;     ///< Energy reflected into upper hemisphere
    float3  absorptionR;    ///< Energy absorbed upper hemisphere
    float3  transmission;   ///< Energy transmitted into upper hemisphere
    float3  absorptionT;    ///< Energy absorbed lower hemisphere

    __init(float3 r, float3 ar, float3 t, float3 at)
    {
        reflection = r;
        absorptionR = ar;
        transmission = t;
        absorptionT = at;
    }
}

AlbedoContributions operator+(const AlbedoContributions a, const AlbedoContributions b)
{
    return AlbedoContributions(
        a.reflection + b.reflection,
        a.absorptionR + b.absorptionR,
        a.transmission + b.transmission,
        a.absorptionT + b.absorptionT );
}

AlbedoContributions operator*(const AlbedoContributions a, const AlbedoContributions b)
{
    return AlbedoContributions(
        a.reflection * b.reflection,
        a.absorptionR * b.absorptionR,
        a.transmission * b.transmission,
        a.absorptionT * b.absorptionT);
}

AlbedoContributions operator*(const float f, const AlbedoContributions b)
{
    return AlbedoContributions(
        f * b.reflection,
        f * b.absorptionR,
        f * b.transmission,
        f * b.absorptionT);
}

AlbedoContributions operator+=(AlbedoContributions a, const AlbedoContributions b)
{
    a.reflection += b.reflection;
    a.absorptionR += b.absorptionR;
    a.transmission += b.transmission;
    a.absorptionT += b.absorptionT;

    return a;
}

struct RoughnessInformation
{
    float2  roughnessBSDFNotation;     ///< This is the (mostly meaningless) non-universal roughness of the BSDF
    float3  scratch;

    __init()
    {
        roughnessBSDFNotation = float2(0.f);
        scratch = float3(0.f);
    }
}

[anyValueSize(96)]
interface IBSDF
{
    /** Evaluates the BSDF.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \param[in,out] sg Sample generator.
        \return Returns f(wi, wo) * dot(wo, n).
    */
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg);

    /** Samples the BSDF.
        \param[in] wi Incident direction.
        \param[out] wo Outgoing direction.
        \param[out] pdf pdf with respect to solid angle for sampling outgoing direction wo (0 if a delta event is sampled).
        \param[out] weight Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
        \param[out] lobeType Sampled lobeType (see LobeType).
        \param[in,out] sg Sample generator.
        \return Returns true if successful.
    */
    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg);

    /** Evaluates the directional pdf for sampling outgoing direction wo.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \return Returns the pdf with respect to solid angle for sampling outgoing direction wo (0 for delta events).
    */
    float evalPdf(const float3 wi, const float3 wo);

    /** Albedo (hemispherical reflectance) of the BSDF. Relfection+transmission hemisphere should be <= 1.0.
        \param[in] wi Incident direction.
        \param[in] lobetype lobe types to be evaluated
        \return Returns the albedo.
    */
    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype);

    /** Information about roughness of the BSDF in (various) forms.
        \param[in] wi Incident direction.
        \return Returns the roughness.
    */
    RoughnessInformation getRoughnessInformation(const float3 wi);
}
