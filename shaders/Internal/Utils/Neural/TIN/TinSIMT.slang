/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#if !defined(FALCOR_ENABLE_NV_COOP_VECTOR) || !defined(FALCOR_ENABLE_TIN)
#error Expected FALCOR_ENABLE_NV_COOP_VECTOR and FALCOR_ENABLE_TIN to be defined
#endif

#if defined(FALCOR_D3D12)
#if !FALCOR_ENABLE_TIN || defined(FALCOR_SIMT_API_DISABLE_TENSOR_CORE)
import TinCommon;
#include "TinSIMTPerThread.slang"
#include "TinSIMTOperators.slang"
#else
import Tin;
#include "TinSIMTCoopD3D12.slang"
#include "TinSIMTOperators.slang"
#endif
#else
import TinCommon;
#if !FALCOR_ENABLE_NV_COOP_VECTOR || defined(FALCOR_SIMT_API_DISABLE_TENSOR_CORE)
#include "TinSIMTPerThread.slang"
#include "TinSIMTOperators.slang"
#else
#include "TinSIMTCoopVulkan.slang"
#endif
#endif

#include "TinSIMTActivations.slang"
