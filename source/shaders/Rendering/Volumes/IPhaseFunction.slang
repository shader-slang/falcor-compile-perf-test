/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;

/** Low-level interface for phase functions.

    Conventions:
     - The incident and outgoing direction point away from the shading location.
     - The incident direction (wi) is given.
     - The outgoing direction (wo) is sampled.
*/
[anyValueSize(36)]
interface IPhaseFunction
{
    /** Evaluates the phase function.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \return Returns f_p(wi, wo).
    */
    float3 eval(const float3 wi, const float3 wo);

    /** Samples the phase function.
        \param[in] wi Incident direction.
        \param[out] wo Outgoing direction.
        \param[out] pdf pdf with respect to solid angle for sampling outgoing direction wo.
        \param[out] weight Sample weight f_p(wi, wo) / pdf(wo).
        \param[in,out] sg Sample generator.
        \return Returns true if successful.
    */
    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg);

    /** Evaluates the phase function pdf for sampling outgoing direction wo.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \return Returns the pdf with respect to solid angle for sampling outgoing direction wo.
    */
    float evalPdf(const float3 wi, const float3 wo);
}
