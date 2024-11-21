/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include <slang.h>
#include <map>

#include "ProgramReflection.h"
#include "Program.h"
#include "ProgramVersion.h"

using namespace slang;

ref<EntryPointGroupReflection> EntryPointGroupReflection::create(
    ProgramVersion const* pProgramVersion,
    uint32_t groupIndex,
    const std::vector<slang::EntryPointLayout*>& pSlangEntryPointReflectors
)
{
    auto pGroup = ref<EntryPointGroupReflection>(new EntryPointGroupReflection(pProgramVersion));
    return pGroup;
}

ref<const ProgramReflection> ProgramReflection::create(
    ProgramVersion const* pProgramVersion,
    slang::ShaderReflection* pSlangReflector,
    const std::vector<slang::EntryPointLayout*>& pSlangEntryPointReflectors,
    std::string& log
)
{
    return ref<const ProgramReflection>(new ProgramReflection(pProgramVersion, pSlangReflector, pSlangEntryPointReflectors, log));
}

ProgramReflection::ProgramReflection(
    ProgramVersion const* pProgramVersion,
    slang::ShaderReflection* pSlangReflector,
    const std::vector<slang::EntryPointLayout*>& pSlangEntryPointReflectors,
    std::string& log
)
    : mpProgramVersion(pProgramVersion), mpSlangReflector(pSlangReflector)
{
    auto pProgram = pProgramVersion->getProgram();

    auto entryPointGroupCount = pProgram->getEntryPointGroupCount();

    for (uint32_t gg = 0; gg < entryPointGroupCount; ++gg)
    {
        ref<EntryPointGroupReflection> pEntryPointGroup =
            EntryPointGroupReflection::create(pProgramVersion, gg, pSlangEntryPointReflectors);
        mEntryPointGroups.push_back(pEntryPointGroup);
    }
}
