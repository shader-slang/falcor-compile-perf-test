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
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/slang.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/slang.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/${FALCOR_SLANG_CONFIG}/libslang.so
    )
endif()

# slang-gfx
if(FALCOR_WINDOWS)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/gfx.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/gfx.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/${FALCOR_SLANG_CONFIG}/libgfx.so
    )
endif()
