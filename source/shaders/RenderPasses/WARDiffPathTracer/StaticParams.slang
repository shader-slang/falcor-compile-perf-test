/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#ifndef SAMPLES_PER_PIXEL
#error RenderPasses/WARDiffPathTracer/StaticParams.slang is missing defines
#endif

__exported import DiffRendering.SharedTypes;

static const uint kSamplesPerPixel = SAMPLES_PER_PIXEL;
static const uint kMaxBounces = MAX_BOUNCES;

static const DiffMode kDiffMode = DiffMode(DIFF_MODE);

static const bool kUseBSDFSampling = USE_BSDF_SAMPLING;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;

static const bool kUseWAR = USE_WAR;
static const uint kAuxSampleCount = AUX_SAMPLE_COUNT;
static const float kLog10vMFConcentration = LOG10_VMF_CONCENTRATION;
static const float kLog10vMFConcentrationScreen = LOG10_VMF_CONCENTRATION_SCREEN;
static const float kBoundaryTermBeta = BOUNDARY_TERM_BETA;
static const bool kUseAntitheticSampling = USE_ANTITHETIC_SAMPLING;
static const float kHarmonicGamma = HARMONIC_GAMMA;

static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
