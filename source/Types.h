#pragma once

// Copy from Falcor/Source/Falcor/Utils/Sampling/SampleGeneratorType.slangh
#define SAMPLE_GENERATOR_TINY_UNIFORM 0



// Copy from Falcor/Source/Falcor/Core/API/Types.h
enum class ShaderType
{
    Vertex,        ///< Vertex shader
    Pixel,         ///< Pixel shader
    Geometry,      ///< Geometry shader
    Hull,          ///< Hull shader (AKA Tessellation control shader)
    Domain,        ///< Domain shader (AKA Tessellation evaluation shader)
    Compute,       ///< Compute shader
    RayGeneration, ///< Ray generation shader
    Intersection,  ///< Intersection shader
    AnyHit,        ///< Any hit shader
    ClosestHit,    ///< Closest hit shader
    Miss,          ///< Miss shader
    Callable,      ///< Callable shader
    Count          ///< Shader Type count
};

enum class ShaderModel
{
    Unknown = 0,
    SM6_0 = 60,
    SM6_1 = 61,
    SM6_2 = 62,
    SM6_3 = 63,
    SM6_4 = 64,
    SM6_5 = 65,
    SM6_6 = 66,
    SM6_7 = 67,
};

// Copy from Falcor/Source/Falcor/Core/API/Raytracing.h
enum class RtPipelineFlags
{
    None = 0,
    SkipTriangles = 0x1,
    SkipProceduralPrimitives = 0x2,
};

enum SlangCompilerFlags: uint32_t
{
    None = 0x0,
    TreatWarningsAsErrors = 0x1,
    /// Enable dumping of intermediate artifacts during compilation.
    /// Note that if a shader is cached no artifacts are being produced.
    /// Delete the `.shadercache` directory in the build directory before dumping.
    DumpIntermediates = 0x2,
    FloatingPointModeFast = 0x4,
    FloatingPointModePrecise = 0x8,
    GenerateDebugInfo = 0x10,
    MatrixLayoutColumnMajor = 0x20, // Falcor is using row-major, use this only to compile stand-alone external shaders.
};

inline constexpr uint32_t getRaytracingMaxAttributeSize()
{
    return 32;
}

inline uint32_t getShaderModelMajorVersion(ShaderModel sm)
{
    return uint32_t(sm) / 10;
}

inline uint32_t getShaderModelMinorVersion(ShaderModel sm)
{
    return uint32_t(sm) % 10;
}

template<typename E>
inline bool is_set(E val, E flag)
{
    return (val & flag) != static_cast<E>(0);
}
