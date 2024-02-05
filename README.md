# falcor-compile-perf-test

## Configure Presets
falcor-compile-perf-test uses _CMake Presets_ store in `CMakePresets.json` to provide a set of commonly used build configurations. You can get the full list of available configure presets running `cmake --list-presets`:

```
$ cmake --list-presets
Available configure presets:

  "windows-vs2022"           - Windows VS2022
  "windows-ninja-msvc"       - Windows Ninja/MSVC
  "linux-clang"              - Linux Ninja/Clang
  "linux-gcc"                - Linux Ninja/GCC
```

Use `cmake --preset <preset name>` to generate the build tree for a given preset. The build tree is written to the `build/<preset name>` folder and the binary output files are in `build/<preset name>/bin`.

An existing build tree can be compiled using `cmake --build build/<preset name>`.

## Run perftest
```
cd build/<preset name>/bin/Debug
./falcor_perftest
```
