:: Copyright 2019-2020 NVIDIA CORPORATION
:: Copyright 2024 The Khronos Group, Inc.

@echo off
setlocal enableextensions

call "%~dp0\packman" init
set "PYTHONPATH=%PM_MODULE_DIR%;%PYTHONPATH%"

if not defined PYTHONNOUSERSITE (
    set PYTHONNOUSERSITE=1
)

REM For performance, default to unbuffered; however, allow overriding via
REM PYTHONUNBUFFERED=0 since PYTHONUNBUFFERED on windows can truncate output
REM when printing long strings
if not defined PYTHONUNBUFFERED (
    set PYTHONUNBUFFERED=1
)

"%PM_PYTHON%" %*