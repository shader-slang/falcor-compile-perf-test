#include "path-tracer.h"

#define FALCOR_INTERNAL 1

enum class SDFGridIntersectionMethod : uint32_t
{
    None = 0,
    GridSphereTracing = 1,
    VoxelSphereTracing = 2,
#ifdef FALCOR_INTERNAL
    VoxelOptimizedSphereTracing = 3,
    VoxelCubicAnalytic = 4,
    VoxelCubicNumericMarmitt = 5,
    VoxelCubicNumericNewtonRaphson = 6,
#endif
};

enum class SDFGridGradientEvaluationMethod : uint32_t
{
    None = 0,
    NumericDiscontinuous = 1,
    NumericContinuous = 2,
#ifdef FALCOR_INTERNAL
    AnalyticDiscontinuous = 3,
    InterpolatedContinuous = 4,
#endif
};

class SDFGrid
{
public:
    enum Type : uint32_t
    {
        None = 0,
        NormalizedDenseGrid = 1,
        SparseVoxelSet = 2,
        SparseBrickSet = 3,
        SparseVoxelOctree = 4,
    };
};

DefineList PathTracer::StaticParams::getDefines(const PathTracer& owner) const
{
    DefineList defines;

    // Path tracer configuration.
    defines.add("MAX_SURFACE_BOUNCES", std::to_string(3));
    defines.add("MAX_DIFFUSE_BOUNCES", std::to_string(maxDiffuseBounces));
    defines.add("MAX_SPECULAR_BOUNCES", std::to_string(maxSpecularBounces));
    defines.add("MAX_TRANSMISSON_BOUNCES", std::to_string(3));
    defines.add("MAX_VOLUME_BOUNCES", std::to_string(maxVolumeBounces));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");

    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", useNEE ? "1" : "0");
    defines.add("USE_MIS", useMIS ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_SCREEN_SPACE_RESTIR", useScreenSpaceReSTIR ? "1" : "0");
    defines.add("USE_NEURAL_RADIANCE_CACHE", useNRC ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("LIMIT_TRANSMISSION", limitTransmission ? "1" : "0");
    defines.add("MAX_TRANSMISSION_REFLECTION_DEPTH", std::to_string(maxTransmissionReflectionDepth));
    defines.add("MAX_TRANSMISSION_REFRACTION_DEPTH", std::to_string(maxTransmissionRefractionDepth));
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("DISABLE_DIRECT_ILLUMINATION", disableDirectIllumination ? "1" : "0");
    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));
    defines.add("DELTA_REFLECTION_CURVATURE_MVECS", deltaReflectionCurvatureMvecs ? "1" : "0");
    defines.add("DELTA_TRANSMISSION_CURVATURE_MVECS", deltaTransmissionCurvatureMvecs ? "1" : "0");
    defines.add("USE_NRD_DEMODULATION", useNRDDemodulation ? "1" : "0");

    // Scheduling configuration.
    defines.add("USE_SPECULAR_QUEUE", useSpecularQueue ? "1" : "0");
    defines.add("PATH_REGENERATION_COUNT_MAIN", std::to_string(usePathRegeneration ? pathRegenerationCountMain : 32u));
    defines.add("PATH_REGENERATION_COUNT_SPECULAR", std::to_string(usePathRegeneration ? pathRegenerationCountSpecular : 32u));
    defines.add("PATH_REGENERATION_COUNT_DELTA_REFLECTION", std::to_string(usePathRegeneration ? pathRegenerationCountDeltaReflection : 32u));
    defines.add("PATH_REGENERATION_COUNT_DELTA_TRANSMISSION", std::to_string(usePathRegeneration ? pathRegenerationCountDeltaTransmission : 32u));
    defines.add("USE_SER", useSER ? "1" : "0");
    defines.add("USE_REORDER_TRACE_RAY_FAST_PATH", useReorderTraceRayFastPath ? "1" : "0");
    defines.add("USE_TRACE_RAY_VISIBILITY", useTraceRayVisibility ? "1" : "0");

    defines.add("USE_ENV_LIGHT", "1");
    defines.add("USE_ANALYTIC_LIGHTS", "0");
    defines.add("USE_EMISSIVE_LIGHTS", "1");
    defines.add("USE_GRID_VOLUMES", "0");
    defines.add("USE_CURVES", "0");
    defines.add("USE_SDF_GRIDS", "0");
    defines.add("USE_HAIR_MATERIAL", "1");

    defines.add("_ACTUAL_MAX_TRIANGLES_PER_NODE", "10");
    defines.add("_DISABLE_NODE_FLUX", "0");
    defines.add("_EMISSIVE_LIGHT_SAMPLER_TYPE", "1");
    defines.add("_SOLID_ANGLE_BOUND_METHOD", "3");
    defines.add("_USE_BOUNDING_CONE", "1");
    defines.add("_USE_LIGHTING_CONE", "1");
    defines.add("_USE_UNIFORM_TRIANGLE_SAMPLING", "1");

    defines.add("MATERIAL_SYSTEM_SAMPLER_DESC_COUNT", std::to_string(256));
    defines.add("MATERIAL_SYSTEM_TEXTURE_DESC_COUNT", std::to_string(105));
    defines.add("MATERIAL_SYSTEM_BUFFER_DESC_COUNT", std::to_string(3));
    defines.add("MATERIAL_SYSTEM_TEXTURE_3D_DESC_COUNT", std::to_string(0));
    defines.add("MATERIAL_SYSTEM_UDIM_INDIRECTION_ENABLED", "0");
    defines.add("MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS", "0");
    defines.add("FALCOR_MATERIAL_INSTANCE_SIZE", std::to_string(128));
    defines.add("FALCOR_INTERNAL", std::to_string(FALCOR_INTERNAL));
    defines.add("FALCOR_ENABLE_TIN", std::to_string(0));
    defines.add("FALCOR_NVAPI_AVAILABLE", std::to_string(0));
    defines.add("FALCOR_ENABLE_NV_COOP_VECTOR", std::to_string(0));

    defines.add("SAMPLE_GENERATOR_TYPE", std::to_string(0));
    defines.add("SAMPLES_PER_PIXEL", std::to_string(1));

    defines.add("GBUFFER_ADJUST_SHADING_NORMALS", std::to_string(1));
    // scene

    defines.add("SCENE_GRID_COUNT", std::to_string(0));
    defines.add("SCENE_HAS_INDEXED_VERTICES", "1");
    defines.add("SCENE_HAS_16BIT_INDICES", "1");
    defines.add("SCENE_HAS_32BIT_INDICES", "0");
    defines.add("SCENE_USE_LIGHT_PROFILE", "0");

    defines.add("SCENE_SDF_GRID_IMPLEMENTATION_NDSDF", std::to_string((uint32_t)SDFGrid::Type::NormalizedDenseGrid));
    defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SVS", std::to_string((uint32_t)SDFGrid::Type::SparseVoxelSet));
    defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SBS", std::to_string((uint32_t)SDFGrid::Type::SparseBrickSet));
    defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SVO", std::to_string((uint32_t)SDFGrid::Type::SparseVoxelOctree));

    defines.add("SCENE_SDF_NO_INTERSECTION_METHOD", std::to_string((uint32_t)SDFGridIntersectionMethod::None));
    defines.add("SCENE_SDF_NO_VOXEL_SOLVER", std::to_string((uint32_t)SDFGridIntersectionMethod::GridSphereTracing));
    defines.add("SCENE_SDF_VOXEL_SPHERE_TRACING", std::to_string((uint32_t)SDFGridIntersectionMethod::VoxelSphereTracing));

#ifdef FALCOR_INTERNAL
    defines.add("SCENE_SDF_VOXEL_OPTIMIZED_SPHERE_TRACING", std::to_string((uint32_t)SDFGridIntersectionMethod::VoxelOptimizedSphereTracing));
    defines.add("SCENE_SDF_VOXEL_CUBIC_SOLVER_ANALYTIC", std::to_string((uint32_t)SDFGridIntersectionMethod::VoxelCubicAnalytic));
    defines.add("SCENE_SDF_VOXEL_CUBIC_SOLVER_NUMERIC_MARMITT", std::to_string((uint32_t)SDFGridIntersectionMethod::VoxelCubicNumericMarmitt));
    defines.add("SCENE_SDF_VOXEL_CUBIC_SOLVER_NUMERIC_NEWTON_RAPHSON", std::to_string((uint32_t)SDFGridIntersectionMethod::VoxelCubicNumericNewtonRaphson));
#endif

    defines.add("SCENE_SDF_NO_GRADIENT_EVALUATION_METHOD", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::None));
    defines.add("SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::NumericDiscontinuous));
    defines.add("SCENE_SDF_GRADIENT_NUMERIC_CONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::NumericContinuous));

#ifdef FALCOR_INTERNAL
    defines.add("SCENE_SDF_GRADIENT_ANALYTIC_DISCONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::AnalyticDiscontinuous));
    defines.add("SCENE_SDF_GRADIENT_INTERPOLATED_CONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::InterpolatedContinuous));
#endif

        // Setup dynamic defines based on current configuration.
    defines.add("SCENE_SDF_GRID_COUNT", std::to_string(0));
    defines.add("SCENE_SDF_GRID_MAX_LOD_COUNT", std::to_string(0));
    //
    defines.add("SCENE_SDF_GRID_IMPLEMENTATION", std::to_string((uint32_t)0));
    defines.add("SCENE_SDF_VOXEL_INTERSECTION_METHOD", std::to_string((uint32_t)0));
    defines.add("SCENE_SDF_GRADIENT_EVALUATION_METHOD", std::to_string((uint32_t)0));
    defines.add("SCENE_SDF_SOLVER_MAX_ITERATION_COUNT", std::to_string(0));
    defines.add("SCENE_SDF_OPTIMIZE_VISIBILITY_RAYS", "0");

    // The following defines may change at runtime.
    defines.add("SCENE_DIFFUSE_ALBEDO_MULTIPLIER", std::to_string(1.f));
    defines.add("SCENE_GEOMETRY_TYPES", std::to_string(2));

    // Grid Volume

    defines.add("GRID_VOLUME_SAMPLER_USE_BRICKEDGRID", std::to_string((uint32_t)1));
    defines.add("GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR", std::to_string((uint32_t)2));
    defines.add("GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER", std::to_string((uint32_t)1));
#ifdef FALCOR_INTERNAL
    defines.add("GRID_VOLUME_SAMPLER_RAY_MARCHING_ROULETTE_THRESHOLD", std::to_string(0.03f));
    defines.add("GRID_VOLUME_SAMPLER_BIASED_RAY_MARCHING_LOOKUP_MULTIPLIER", std::to_string(0.5f));
    defines.add("GRID_VOLUME_SAMPLER_BIASED_RAY_MARCHING_LOCAL_CDF_STEP_SIZE", std::to_string(1.0f));
    defines.add("GRID_VOLUME_SAMPLER_UNBIASED_RAY_MARCHING_LOOKUP_MULTIPLIER", std::to_string(1.0f));
    defines.add("GRID_VOLUME_SAMPLER_UNBIASED_RAY_MARCHING_MAX_ORDER", std::to_string(12));
#endif

    defines.add("HIT_INFO_DEFINES", "1");
    defines.add("HIT_INFO_USE_COMPRESSION", "0");
    defines.add("HIT_INFO_TYPE_BITS", std::to_string(3));
    defines.add("HIT_INFO_INSTANCE_ID_BITS", std::to_string(4));
    defines.add("HIT_INFO_PRIMITIVE_INDEX_BITS", std::to_string(11));

    // Set default (off) values for additional features.
    // The passes that use these will set them at runtime. These don't modify resource declarations.
    defines.add("USE_VIEW_DIR", "0");
    defines.add("OUTPUT_GUIDE_DATA", "0");
    defines.add("OUTPUT_NRD_DATA", "0");
    defines.add("OUTPUT_NRD_ADDITIONAL_DATA", "0");
    defines.add("OUTPUT_TIME", "0");
    defines.add("OUTPUT_DEBUG", "0");
    defines.add("ENABLE_STATS", "0");

    // Stochastic texture filtering specialization
    defines.add("USE_STOCHASTIC_BICUBIC_FILTERING", useStochasticBicubicFiltering ? "1" : "0");
    defines.add("USE_SPATIO_TEMP_BLUE_NOISE", useSpatioTemporalBlueNoise ? "1" : "0");

    return defines;
}
