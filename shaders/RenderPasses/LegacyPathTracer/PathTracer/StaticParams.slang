/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Translation of defines set by the host to compile-time constants used to
    configure the path tracer without overhead from dynamic control flow.
    This will eventually be replaced by specialization constants in Slang.

    The host sets the following defines (booleans are 1=true, 0=false):

    SAMPLES_PER_PIXEL               Number of paths to trace per pixel.
    LIGHT_SAMPLES_PER_VERTEX        Number of light sampler per path vertex.
    MAX_BOUNCES                     Maximum number of indirect bounces (0 means no indirect).
    MAX_NON_SPECULAR_BOUNCES        Maximum number of non-specular indirect bounces (0 means no indirect).
    USE_VBUFFER                     Use a V-buffer as input.
    ADJUST_SHADING_NORMALS          Adjust shading normals on secondary hits.
    FORCE_ALPHA_ONE                 Force the alpha channel to 1.0.
    USE_ANALYTIC_LIGHTS             Enables Falcor's analytic lights (point, directional).
    USE_EMISSIVE_LIGHTS             Enables use of emissive geometry as light sources.
    USE_EMISSIVE_SAMPLER            True if the emissive light sampler should be used.
    USE_ENV_LIGHT                   True if env map is available and should be used as light source.
    USE_ENV_BACKGROUND              True if env map is available and should be used as background.
    USE_BRDF_SAMPLING               Enables BRDF importance sampling.
    USE_MIS                         Enables multiple importance sampling.
    USE_RUSSIAN_ROULETTE            Enables Russian roulette for path termination.
    MIS_HEURISTIC                   MIS heuristic enum value.
    USE_NESTED_DIELECTRICS          Enables nested dielectrics handling.
    USE_LIGHTS_IN_DIELECTRIC_VOLUMES Enables lights in dielectric volumes.
    DISABLE_CAUSTICS                Disables caustics.
*/

#ifndef SAMPLES_PER_PIXEL
#error StaticParams.slang is missing defines
#endif

static const uint kSamplesPerPixel = SAMPLES_PER_PIXEL;
static const uint kLightSamplesPerVertex = LIGHT_SAMPLES_PER_VERTEX;
static const uint kMaxBounces = MAX_BOUNCES;
static const uint kMaxNonSpecularBounces = MAX_NON_SPECULAR_BOUNCES;
static const bool kUseVBuffer = USE_VBUFFER;
static const bool kUseAlphaTest = USE_ALPHA_TEST;
static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
static const bool kForceAlphaOne = FORCE_ALPHA_ONE;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const bool kUseBRDFSampling = USE_BRDF_SAMPLING;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;
static const bool kUseRussianRoulette = USE_RUSSIAN_ROULETTE;
static const uint kMISHeuristic = MIS_HEURISTIC;
// static const MISHeuristic kMISHeuristic = (MISHeuristic) MIS_HEURISTIC; // TODO: Use enum instead of uint when Slang supports it.
static const bool kUseNestedDielectrics = USE_NESTED_DIELECTRICS;

// TODO: https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/886
// Temporary workaround for a driver bug that needs further investigation.
#ifdef MEGAKERNEL_DRIVER_BUG_WAR
static const bool kUseLightsInDielectricVolumes = true;
#else
static const bool kUseLightsInDielectricVolumes = USE_LIGHTS_IN_DIELECTRIC_VOLUMES;
#endif

static const bool kDisableCaustics = DISABLE_CAUSTICS;
