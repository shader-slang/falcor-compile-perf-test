/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import GeneratedModule;

uint test(uint x)
{
    // The function f() is defined in the generated module.
    return f(x);
}
