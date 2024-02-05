add_executable(falcor_perftest)

target_sources(falcor_perftest PRIVATE
    main.cpp
    Object.cpp
    path-tracer.cpp
    Program.cpp
    ProgramManager.cpp
    ProgramReflection.cpp
    ProgramVersion.cpp
    DeviceWrapper.cpp
)

target_copy_shaders(falcor_perftest ./shaders .)

target_compile_features(falcor_perftest
    PUBLIC
        cxx_std_17
    PRIVATE
        cxx_std_17
)

target_compile_options(falcor_perftest
    PUBLIC
        # MSVC flags.
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
            /Zi                             # generate debug symbols
            /MP                             # enable multi-processor compilation
            # Configure warnings
            /WX                             # warnings as errors
            /W4                             # increase warning level
            /wd4819                         # the file contains a character that cannot be represented in the current code page(936)
            /wd4251                         # 'type' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
            /wd4244                         # 'conversion' conversion from 'type1' to 'type2', possible loss of data
            /wd4267                         # 'var' : conversion from 'size_t' to 'type', possible loss of data
            /wd4100                         # unreferenced formal parameter
            /wd4201                         # nonstandard extension used: nameless struct/union
            /wd4245                         # conversion from 'type1' to 'type2', signed/unsigned mismatch
            /wd4189                         # local variable is initialized but not referenced
            /wd4127                         # conditional expression is constant
            /wd4701                         # potentially uninitialized local variable 'name' used
            /wd4703                         # potentially uninitialized local pointer variable 'name' used
            /wd4324                         # structure was padded due to alignment specifier
            /wd4505                         # unreferenced local function has been removed
            /wd4702                         # unreachable code
            /wd4389                         # signed/unsigned mismatch
            /wd4459                         # declaration of 'identifier' hides global declaration
            /wd4268                         # 'identifier' : 'const' static/global data initialized with compiler generated default constructor fills the object with zeros
            /external:templates-            # allows warnings from system headers when they occur in a template that's instantiated in your code
            # Enable warnings that are disabled with /W4
            /we4263                         # 'function': member function does not override any base class virtual member function
            /we4264                         # 'virtual_function': no override available for virtual member function from base 'class'; function is hidden
            /we5038                         # data member 'member1' will be initialized after data member 'member2'
            /we5259                         # 'specialized-type': explicit specialization requires 'template <>'
            /we5263                         # calling 'std::move' on a temporary object prevents copy elision
        >
        # Clang/GCC flags.
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
            -fms-extensions                 # enable MS extensions (among other things allow anonymous structs)
            -fvisibility=hidden             # hide symbols by default
            -Wall                           # set warning level
            -Wno-unused-function
            -Wno-unused-variable
            -Wno-unused-but-set-variable
            -Wno-switch
            -Wno-missing-braces
            -Wno-invalid-offsetof
        >
        # Clang flags.
        $<$<CXX_COMPILER_ID:Clang>:
            -Wno-unused-private-field
            -Wno-braced-scalar-init
            -Wno-self-assign-overloaded
        >
        # GCC flags.
        $<$<CXX_COMPILER_ID:GNU>:
            -fpermissive
            -Wno-sign-compare
            -Wno-literal-suffix
            -Wno-class-memaccess
            -Wno-strict-aliasing
            -Wno-maybe-uninitialized
            -Wno-stringop-truncation
        >
    PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>  # big object files
    )

target_link_options(falcor_perftest
    PUBLIC
        # MSVC flags.
        $<$<CXX_COMPILER_ID:MSVC>:/DEBUG>           # generate debug information
)

target_compile_definitions(falcor_perftest
    PUBLIC
        $<$<CONFIG:Release>:NDEBUG>
        $<$<CONFIG:Debug>:_DEBUG>
        # Windows.
        $<$<PLATFORM_ID:Windows>:NOMINMAX>  # do not define min/max macros
        $<$<PLATFORM_ID:Windows>:UNICODE>   # force character map to unicode
        # MSVC C++ library.
        $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
        $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING>
        $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS>
        # Clang.
        $<$<CXX_COMPILER_ID:Clang>:_MSC_EXTENSIONS> # enable MS extensions
        $<$<PLATFORM_ID:Windows>:WINDOWS>
        $<$<PLATFORM_ID:Linux>:Linux>
    PRIVATE
        PROJECT_DIR="${CMAKE_SOURCE_DIR}/"
)

target_link_libraries(falcor_perftest
    PUBLIC
        slang slang-gfx
        external_includes
)

set_target_properties(falcor_perftest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${FALCOR_LIBRARY_OUTPUT_DIRECTORY}
    SKIP_BUILD_RPATH TRUE)
