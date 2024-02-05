#pragma once
#include "DefineList.h"
#include "Program.h"
#include "Types.h"

// Copy from Falcor/Source/Internal/RenderPasses/InternalPathTracer/Params.slang
enum class MISHeuristic : uint32_t
{
    Balance     = 0,    ///< Balance heuristic.
    PowerTwo    = 1,    ///< Power heuristic (exponent = 2.0).
    PowerExp    = 2,    ///< Power heuristic (variable exponent).
};

enum class ColorFormat : uint32_t
{
    RGBA32F         = 0,
    LogLuvHDR       = 1,
};

enum class SchedulingMode
{
    QueueInline,        ///< Persistent threads using queues and TraceRayInline.
    SimpleInline,       ///< Simple compute pass using TraceRayInline.
    SimpleTraceRay,     ///< Simple raytracing pass using TraceRay.
    ReorderTraceRay,    ///< Raytracing pass using SER/HitObject API.
    ReorderInline,      ///< Raytracing pass using SER/HitObject API and TraceRayInline.
};

// Copy from Falcor/Source/Falcor/Rendering/Lights/EmissiveLightSamplerType.slangh
enum class EmissiveLightSamplerType : uint32_t
{
    Uniform     = 0,
    LightBVH    = 1,
    Power       = 2,

    Null        = 0xff,
};

// Copy from Falcor/Source/Falcor/Rendering/Materials/TexLODTypes.slang
enum class TexLODMode : uint32_t
{
    Mip0 = 0,
    RayCones = 1,
    RayDiffs = 2,
    Stochastic = 3,
};

class PathTracer
{
public:
    /** Static configuration. Changing any of these options require shader recompilation.
    */
    struct StaticParams
    {
        // Rendering parameters
        uint32_t    samplesPerPixel = 1;                        ///< Number of samples (paths) per pixel, unless a sample density map is used.
        uint32_t    maxSurfaceBounces = 0;                      ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLenth. This will be initialized at startup.
        uint32_t    maxDiffuseBounces = 3;                      ///< Max number of diffuse bounces (0 = direct only), up to kMaxBounces.
        uint32_t    maxSpecularBounces = 3;                     ///< Max number of specular bounces (0 = direct only), up to kMaxBounces.
        uint32_t    maxTransmissionBounces = 10;                ///< Max number of transmission bounces (0 = none), up to kMaxBounces.
        uint32_t    maxVolumeBounces = 30;                      ///< Max number of volume bounces (0 = direct only), up to kMaxBounces.

        // Sampling parameters
        uint32_t    sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM; ///< Pseudorandom sample generator type.
        bool        useBSDFSampling = true;                     ///< Use BRDF importance sampling, otherwise cosine-weighted hemisphere sampling.
        bool        useRussianRoulette = false;                 ///< Use russian roulette to terminate low throughput paths.
        bool        useNEE = true;                              ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        bool        useMIS = true;                              ///< Use multiple importance sampling (MIS) when NEE is enabled.
        MISHeuristic misHeuristic = MISHeuristic::Balance;      ///< MIS heuristic.
        float       misPowerExponent = 2.f;                     ///< MIS exponent for the power heuristic. This is only used when 'PowerExp' is chosen.
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::LightBVH;  ///< Emissive light sampler to use for NEE.
        bool        useScreenSpaceReSTIR = false;               ///< Use screen space ReSTIR for direct illumination.
        bool        useNRC = false;                             ///< Use the Neural Radiance Cache for global illumination at last path vertex.
        bool        useDOF = true;                              ///< Can explicitly disable DOF even for non-zero aperture cameras, used when DOF is added in post-process

        // Material parameters
        bool        useAlphaTest = true;                        ///< Use alpha testing on non-opaque triangles.
        bool        adjustShadingNormals = false;               ///< Adjust shading normals on secondary hits.
        uint32_t    maxNestedMaterials = 2;                     ///< Maximum supported number of nested materials.
        bool        useLightsInDielectricVolumes = false;       ///< Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.
        bool        limitTransmission = false;                  ///< Limit specular transmission by handling reflection/refraction events only up to a given transmission depth.
        uint32_t    maxTransmissionReflectionDepth = 0;         ///< Maximum transmission depth at which to sample specular reflection. Only used if limitTransmission is true.
        uint32_t    maxTransmissionRefractionDepth = 0;         ///< Maximum transmission depth at which to sample specular refraction (after that, IoR is set to 1). Only used if limitTransmission is true.
        bool        disableCaustics = false;                    ///< Disable sampling of caustics.
        bool        disableDirectIllumination = false;          ///< Disable all direct illumination.
        TexLODMode  primaryLodMode = TexLODMode::Mip0;          ///< Use filtered texture lookups at the primary hit.

        // Stochastic texture filtering parameters
        bool        useSpatioTemporalBlueNoise = true;          ///< Use spatio-temporal blue noise
        bool        useStochasticBicubicFiltering = false;      ///< Use B-spline bicubic filtering

        // Output parameters
        ColorFormat colorFormat = ColorFormat::LogLuvHDR;       ///< Color format used for internal per-sample color and denoiser buffers.

        // Denoising parameters
        bool        deltaReflectionCurvatureMvecs = true;       ///< Take normal curvature into account when generating reflection motion vectors.
        bool        deltaTransmissionCurvatureMvecs = false;    ///< Take normal curvature into account when generating transmission motion vectors (keep disabled by default since it only works for simple shapes for now).
        bool        useNRDDemodulation = true;                  ///< Global switch for NRD demodulation.

        // Scheduling parameters
        SchedulingMode schedulingMode = SchedulingMode::QueueInline;///< Scheduling mode.
        bool        useSpecularQueue = true;                        ///< Enable processing of specular paths on a separate queue.
        bool        usePathRegeneration = true;                     ///< Generate new paths as paths terminate, instead of waiting until warp is done.
        uint32_t    pathRegenerationCountMain = 32;                 ///< Wait until this many threads are available before generating new paths on the main queue. 32 effectively disables path regeneration.
        uint32_t    pathRegenerationCountSpecular = 24;             ///< Wait until this many threads are available before generating new paths on the specular queue. 32 effectively disables path regeneration.
        uint32_t    pathRegenerationCountDeltaReflection = 24;      ///< Wait until this many threads are available before generating new paths on the delta reflection queue. 32 effectively disables path regeneration.
        uint32_t    pathRegenerationCountDeltaTransmission = 24;    ///< Wait until this many threads are available before generating new paths on the delta transmission queue. 32 effectively disables path regeneration.
        bool        useSER = true;                                  ///< Enable using SER for reordering schedulers.
        bool        useReorderTraceRayFastPath = true;              ///< Use fast path with reduced continuation spills for the ReorderTraceRay scheduler.
        bool        useTraceRayVisibility = false;                  ///< Use TraceRay visibility queries for the ReorderTraceRay scheduler.

        DefineList getDefines(const PathTracer& owner) const;
        TypeConformanceList getTypeConformances(const PathTracer& owner) const;
    };

    StaticParams    m_staticParams;
};
