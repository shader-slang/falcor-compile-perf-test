/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.TexLODTypes;

/** Translation of defines set by the host to compile-time constants used to
    configure the path tracer without overhead from dynamic control flow.
    This will eventually be replaced by specialization constants in Slang.
*/

#ifndef SAMPLES_PER_PIXEL
#error StaticParams.slang is missing defines
#endif

static const uint kUseViewDir = USE_VIEW_DIR;
static const uint kSamplesPerPixel = SAMPLES_PER_PIXEL;
static const uint kMaxSurfaceBounces = MAX_SURFACE_BOUNCES;
static const uint kMaxDiffuseBounces = MAX_DIFFUSE_BOUNCES;
static const uint kMaxSpecularBounces = MAX_SPECULAR_BOUNCES;
static const uint kMaxTransmissionBounces = MAX_TRANSMISSON_BOUNCES;
static const uint kMaxVolumeBounces = MAX_VOLUME_BOUNCES;
static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
static const bool kUseBSDFSampling = USE_BSDF_SAMPLING;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;
static const bool kUseRussianRoulette = USE_RUSSIAN_ROULETTE;
static const bool kUseScreenSpaceReSTIR = USE_SCREEN_SPACE_RESTIR;
static const bool kUseNRC = USE_NEURAL_RADIANCE_CACHE;
static const bool kUseAlphaTest = USE_ALPHA_TEST;
static const bool kUseLightsInDielectricVolumes = USE_LIGHTS_IN_DIELECTRIC_VOLUMES;
static const bool kLimitTransmission = LIMIT_TRANSMISSION;
static const uint kMaxTransmissionReflectionDepth = MAX_TRANSMISSION_REFLECTION_DEPTH;
static const uint kMaxTransmissionRefractionDepth = MAX_TRANSMISSION_REFRACTION_DEPTH;
static const bool kDisableCaustics = DISABLE_CAUSTICS;
static const bool kDisableDirectIllumination = DISABLE_DIRECT_ILLUMINATION;
static const TexLODMode kPrimaryLodMode = TexLODMode(PRIMARY_LOD_MODE);
static const uint kColorFormat = COLOR_FORMAT;
static const uint kMISHeuristic = MIS_HEURISTIC;
static const float kMISPowerExponent = MIS_POWER_EXPONENT;
static const bool kDeltaReflectionCurvatureMvecs = DELTA_REFLECTION_CURVATURE_MVECS;
static const bool kDeltaTransmissionCurvatureMvecs = DELTA_TRANSMISSION_CURVATURE_MVECS;
static const bool kUseNRDDemodulation = USE_NRD_DEMODULATION;
