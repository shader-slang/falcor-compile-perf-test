cmake_minimum_required(VERSION 3.15)

cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0057 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

project(Falcor
    DESCRIPTION "Falcor Perf-Test"
    LANGUAGES CXX C
)

# -----------------------------------------------------------------------------
# Check platform
# -----------------------------------------------------------------------------

if(${CMAKE_SYSTEM_NAME} MATCHES "Window")
    set(FALCOR_PLATFORM "Windows")
    set(FALCOR_WINDOWS TRUE)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(FALCOR_PLATFORM "Linux")
    set(FALCOR_LINUX TRUE)
else()
    message(FATAL_ERROR "Unsupported platform!")
endif()
message(STATUS "Platform: ${FALCOR_PLATFORM}")

# -----------------------------------------------------------------------------
# Packman
# -----------------------------------------------------------------------------

# Falcor uses packman to pull binary dependencies. We need to pull the dependencies
# before CMake starts configuring the project as some of the configuration relies
# on these dependencies being available. We also add additional targets to pull
# the dependencies when the project is built such that they are updated automatically
# if the manifest files change.

if(FALCOR_WINDOWS)
    set(PACKMAN "${CMAKE_SOURCE_DIR}/tools/packman/packman.cmd")
    set(PACKMAN_PLATFORM "windows-x86_64")
elseif(FALCOR_LINUX)
    set(PACKMAN "${CMAKE_SOURCE_DIR}/tools/packman/packman")
    set(PACKMAN_PLATFORM "linux-x86_64")
endif()

# Pull dependencies at configure time.
message(STATUS "Updating packman dependencies ...")
execute_process(
    COMMAND ${PACKMAN} pull ${CMAKE_SOURCE_DIR}/dependencies.xml --platform ${PACKMAN_PLATFORM}
    COMMAND_ERROR_IS_FATAL ANY
)

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------

# Require builds to be outside of source tree.
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please use a build directory instead.")
endif()

# Enable folders (for Visual Studio).
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Setup available build configurations.
if(NOT SETUP_CONFIGURATION_TYPES)
    set(SETUP_CONFIGURATION_TYPES 1)

    if(CMAKE_CONFIGURATION_TYPES)
        # multi config generator
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
    else()
        # single config generator
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
    endif()
endif()

set(FALCOR_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(FALCOR_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_CONFIGURATION_TYPES)
    set(FALCOR_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>)
else()
    set(FALCOR_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY})
endif()

set(FALCOR_SHADER_OUTPUT_DIRECTORY ${FALCOR_OUTPUT_DIRECTORY}/shaders)

# # -----------------------------------------------------------------------------
# # External dependencies
# # -----------------------------------------------------------------------------
#
add_subdirectory(external)

#
# -----------------------------------------------------------------------------
# Packman dependencies
# -----------------------------------------------------------------------------

add_custom_target(packman_dependencies DEPENDS packman_dependencies_stamp)
set_target_properties(packman_dependencies PROPERTIES FOLDER "Misc")
add_custom_command(
    OUTPUT packman_dependencies_stamp
    COMMAND ${PACKMAN} pull ${CMAKE_SOURCE_DIR}/dependencies.xml --platform ${PACKMAN_PLATFORM}
    COMMAND ${CMAKE_COMMAND} -E touch packman_dependencies_stamp
    MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/dependencies.xml
    COMMENT "Updating packman dependencies"
)

# -----------------------------------------------------------------------------
# Shader file handling
# -----------------------------------------------------------------------------

# Setup build rules to copy all shaders of a target to the output directory.
# The specified output_dir is relative to the global shader output directory (FALCOR_SHADER_OUTPUT_DIRECTORY).
function(target_copy_shaders target source_dir output_dir)

    get_target_property(target_source_dir ${target} SOURCE_DIR)

    # Create custom commands for copying shader sources.
    set(src_file ${target_source_dir}/${source_dir})

    add_custom_command(
        TARGET ${target}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${src_file} ${FALCOR_SHADER_OUTPUT_DIRECTORY}
        COMMENT "${target}: Copying shader ${source_dir}"
    )
endfunction()

# -----------------------------------------------------------------------------
# Project sources
# -----------------------------------------------------------------------------

add_subdirectory(source)
