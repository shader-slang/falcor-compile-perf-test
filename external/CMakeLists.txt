# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------

# external_includes
add_library(external_includes INTERFACE)
target_include_directories(external_includes INTERFACE include)
# -----------------------------------------------------------------------------
# Dependencies in packman
# -----------------------------------------------------------------------------

set(PACKMAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packman)

# -----------------------------------------------------------------------------
# Pre-built dependencies in packman
# -----------------------------------------------------------------------------

# slang
set(SLANG_DIR ${PACKMAN_DIR}/slang)
set(FALCOR_SLANG_CONFIG release CACHE STRING "Slang config to use.")
set_property(CACHE FALCOR_SLANG_CONFIG PROPERTY STRINGS release debug)
if(FALCOR_WINDOWS)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}/include
        IMPORTED_IMPLIB ${SLANG_DIR}/lib/slang.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/slang.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}/include
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/${FALCOR_SLANG_CONFIG}/libslang.so
    )
endif()

# slang-gfx
if(FALCOR_WINDOWS)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}/include
        IMPORTED_IMPLIB ${SLANG_DIR}/lib/gfx.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/gfx.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}/include
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/${FALCOR_SLANG_CONFIG}/libgfx.so
    )
endif()

set(VULKAN_DIR ${PACKMAN_DIR}/vulkan/Vulkan-Headers-1.4.310)
target_include_directories(external_includes INTERFACE ${VULKAN_DIR}/include)
