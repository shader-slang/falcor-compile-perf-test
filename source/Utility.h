#pragma once

#include <algorithm>

#define PATH_MAX 1024

#define ASSERT(x)               \
    do {                        \
        if (!(x)) {              \
            printf("^^^^Assert FAILED: __file__: %s __func__: %s, __LINE__: %d\n", __FILE__, __func__, __LINE__); \
            assert(0);            \
        }                       \
    } while(0)

#define ASSERT_EQ(act, exp, msg)    \
    if ((act) != (exp)) {           \
        printf("^^^^Assert FAILED: __file__: %s __func__: %s, __LINE__: %d\n", __FILE__, __func__, __LINE__); \
        assert(0);                  \
    }                               \
    else {                          \
        printf("%s Pass\n", msg);   \
    }


inline SlangStage getSlangStage(ShaderType type)
{
    switch (type)
    {
    case ShaderType::Vertex:
        return SLANG_STAGE_VERTEX;
    case ShaderType::Pixel:
        return SLANG_STAGE_PIXEL;
    case ShaderType::Geometry:
        return SLANG_STAGE_GEOMETRY;
    case ShaderType::Hull:
        return SLANG_STAGE_HULL;
    case ShaderType::Domain:
        return SLANG_STAGE_DOMAIN;
    case ShaderType::Compute:
        return SLANG_STAGE_COMPUTE;
    case ShaderType::RayGeneration:
        return SLANG_STAGE_RAY_GENERATION;
    case ShaderType::Intersection:
        return SLANG_STAGE_INTERSECTION;
    case ShaderType::AnyHit:
        return SLANG_STAGE_ANY_HIT;
    case ShaderType::ClosestHit:
        return SLANG_STAGE_CLOSEST_HIT;
    case ShaderType::Miss:
        return SLANG_STAGE_MISS;
    case ShaderType::Callable:
        return SLANG_STAGE_CALLABLE;
    default:
        assert(!"Unreachable");
        return SLANG_STAGE_NONE;
    }
}

#if defined(Linux)
#include <filesystem>
#include <unistd.h>
inline const std::filesystem::path& getExecutablePath()
{
    static std::filesystem::path path(
        []()
        {
            char pathStr[PATH_MAX] = {0};
            if (readlink("/proc/self/exe", pathStr, PATH_MAX) == -1)
            {
                assert(!"Failed to get the executable path.");
            }
            return std::filesystem::path(pathStr);
        }()
    );
    return path;
}
#elif defined(_WIN32)
#include <windows.h>
inline const std::filesystem::path& getExecutablePath()
{
    static std::filesystem::path path(
        []()
        {
            CHAR pathStr[1024];
            if (GetModuleFileNameA(nullptr, pathStr, ARRAYSIZE(pathStr)) == 0)
            {
                assert(!"Failed to get the executable path.");
            }
            return std::filesystem::path(pathStr);
        }()
    );
    return path;
}
#else
#error "No OS specified"
#endif

inline std::string getSlangProfileString(ShaderModel shaderModel)
{
    char buffer[80];
    std::snprintf(buffer, sizeof(buffer), "sm_%u_%u", getShaderModelMajorVersion(shaderModel), getShaderModelMinorVersion(shaderModel));
    return std::string(buffer);
}

inline std::vector<std::filesystem::path> getInitialShaderDirectories()
{
    static std::filesystem::path directory = getExecutablePath().parent_path() / "shaders";
    printf("shader path = %s\n", directory.string().c_str());

    std::vector<std::filesystem::path> developmentDirectories = { directory };

    return developmentDirectories;
}

static std::vector<std::filesystem::path> gShaderDirectories = getInitialShaderDirectories();

inline const std::vector<std::filesystem::path>& getShaderDirectoriesList()
{
    return gShaderDirectories;
}

inline std::string getExtensionFromPath(const std::filesystem::path& path)
{
    std::string ext;
    if (path.has_extension())
    {
        ext = path.extension().string();
        // Remove the leading '.' from ext.
        if (ext.size() > 0 && ext[0] == '.')
            ext.erase(0, 1);
        // Convert to lower-case.
        std::transform(ext.begin(), ext.end(), ext.begin(), [](char c) { return std::tolower(c); });
    }
    return ext;
}

inline bool hasExtension(const std::filesystem::path& path, std::string_view ext)
{
    // Remove leading '.' from ext.
    if (ext.size() > 0 && ext[0] == '.')
        ext.remove_prefix(1);

    std::string pathExt = getExtensionFromPath(path);

    if (ext.size() != pathExt.size())
        return false;

    return std::equal(ext.rbegin(), ext.rend(), pathExt.rbegin(), [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

inline bool findFileInShaderDirectories(const std::filesystem::path& path, std::filesystem::path& fullPath)
{
    // Check if this is an absolute path.
    if (path.is_absolute())
    {
        if (std::filesystem::exists(path))
        {
            fullPath = std::filesystem::canonical(path);
            return true;
        }
    }

    // Search in other paths.
    for (const auto& dir : gShaderDirectories)
    {
        fullPath = dir / path;
        if (std::filesystem::exists(fullPath))
        {
            fullPath = std::filesystem::canonical(fullPath);
            return true;
        }
    }

    return false;
}

inline void DumpArguments(slang::SessionDesc& sessionDesc, std::vector<const char*> args)
{
    printf("targetCount = %lld\n", sessionDesc.targetCount);

    for (uint32_t i = 0; i < sessionDesc.targetCount; i++)
    {
        printf("targets.format = %d\n", sessionDesc.targets[i].format);
        printf("targets.profile = %d\n", sessionDesc.targets[i].profile);
        printf("targets.flags = %d\n", sessionDesc.targets[i].flags);
        printf("targets.floatingPointMode = %d\n", sessionDesc.targets[i].floatingPointMode);
        printf("targets.lineDirectiveMode = %d\n", sessionDesc.targets[i].lineDirectiveMode);
        printf("targets.forceGLSLScalarBufferLayout = %d\n", sessionDesc.targets[i].forceGLSLScalarBufferLayout);
    }

    printf("flags = %d\n", sessionDesc.flags);
    printf("defaultMatrixLayoutMode = %d\n", sessionDesc.defaultMatrixLayoutMode);
    printf("searchPathCount = %lld\n", sessionDesc.searchPathCount);
    for (uint32_t i = 0; i < sessionDesc.searchPathCount; i++)
        printf("searchPaths = %s\n", sessionDesc.searchPaths[i]);

    printf("preprocessorMacroCount = %lld\n", sessionDesc.preprocessorMacroCount);
    for (uint32_t i = 0; i < sessionDesc.preprocessorMacroCount; i++)
        printf("macro [name = %s, value = %s]\n", sessionDesc.preprocessorMacros[i].name, sessionDesc.preprocessorMacros[i].value);

    for (uint32_t i = 0; i < args.size(); i++) {
        printf("%s\n", args[i]);
    }
}
