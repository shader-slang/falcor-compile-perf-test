/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;

 /** Interface for NDFs (normal distribution functions) from microfacet theory.
 
     These are assumed to be "stretch-invariant", which means only a few methods
     need to be implemented for evaluating and sampling from the underlying
     distribution of slopes.
     In particular, it is sufficient to characterize the slope distribution in
     a "standard configuration", meaning it is isotropic with a fixed unit roughness
     of alpha = 1.0. This then allows derivation of more general anisotropic variants
     and accurate Smith masking functions.
 
     See "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs" by
     Eric Heitz 2014 for more details.
 */
 interface INDF
 {
     /** Evaluates the slope distribution in the standard configuration.
 
         \param[in] slope 2D slope.
         \return P22_std(slope).
     */
     float evalP22Std(float2 slope);
 
     /** Evaluates the projected area (sigma) of the heighfield in the standard configuration.
 
         \param[in] dir Evaluated direction in local space, stretched into the standard configuration.
         \return sigma_std(dir).
     */
     float evalSigmaStd(float3 dir);
 
     /** Samples a slope proportionally to P22_std(slope).
 
         \param[in,out] sg Sample generator.
         \return Sampled slope.
     */
     float2 sampleP22Std<S : ISampleGenerator>(inout S sg);
     
     /** Samples a slope proportionally to only the visible P22_std(slope).
 
         \param[in] wi Incident direction, stretched into the standard configuration.
         \param[in,out] sg Sample generator.
         \return Sampled slope.
     */
     float2 sampleVisibleP22Std<S : ISampleGenerator>(float3 wi, inout S sg);
 }
 