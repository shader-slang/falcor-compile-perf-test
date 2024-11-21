/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include <set>
#include <slang.h>

#include "ProgramVersion.h"
#include "Program.h"
#include "Utility.h"


ref<const EntryPointGroupKernels> EntryPointGroupKernels::create(
    EntryPointGroupKernels::Type type,
    const std::vector<ref<EntryPointKernel>>& kernels,
    const std::string& exportName
)
{
    return ref<EntryPointGroupKernels>(new EntryPointGroupKernels(type, kernels, exportName));
}

EntryPointGroupKernels::EntryPointGroupKernels(Type type, const std::vector<ref<EntryPointKernel>>& kernels, const std::string& exportName)
    : mType(type), mKernels(kernels), mExportName(exportName)
{}

const EntryPointKernel* EntryPointGroupKernels::getKernel(ShaderType type) const
{
    for (auto& pKernel : mKernels)
    {
        if (pKernel->getType() == type)
            return pKernel.get();
    }
    return nullptr;
}

//
// ProgramKernels
//

ProgramKernels::ProgramKernels(
    const ProgramVersion* pVersion,
    const ref<const ProgramReflection>& pReflector,
    const ProgramKernels::UniqueEntryPointGroups& uniqueEntryPointGroups,
    const std::string& name
)
    : mName(name), mUniqueEntryPointGroups(uniqueEntryPointGroups), mpReflector(pReflector), mpVersion(pVersion)
{}

ref<ProgramKernels> ProgramKernels::create(
    Device* pDevice,
    const ProgramVersion* pVersion,
    slang::IComponentType* pSpecializedSlangGlobalScope,
    const std::vector<slang::IComponentType*>& pTypeConformanceSpecializedEntryPoints,
    const ref<const ProgramReflection>& pReflector,
    const ProgramKernels::UniqueEntryPointGroups& uniqueEntryPointGroups,
    std::string& log,
    const std::string& name
)
{
    ref<ProgramKernels> pProgram = ref<ProgramKernels>(new ProgramKernels(pVersion, pReflector, uniqueEntryPointGroups, name));

    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.linkingStyle = gfx::IShaderProgram::LinkingStyle::SeparateEntryPointCompilation;
    programDesc.slangGlobalScope = pSpecializedSlangGlobalScope;

    // Check if we are creating program kernels for ray tracing pipeline.
    bool isRayTracingProgram = false;
    if (pTypeConformanceSpecializedEntryPoints.size())
    {
        auto stage = pTypeConformanceSpecializedEntryPoints[0]->getLayout()->getEntryPointByIndex(0)->getStage();
        switch (stage)
        {
        case SLANG_STAGE_ANY_HIT:
        case SLANG_STAGE_RAY_GENERATION:
        case SLANG_STAGE_CLOSEST_HIT:
        case SLANG_STAGE_CALLABLE:
        case SLANG_STAGE_INTERSECTION:
        case SLANG_STAGE_MISS:
            isRayTracingProgram = true;
            break;
        default:
            break;
        }
    }
    // Deduplicate entry points by name for ray tracing program.
    std::vector<slang::IComponentType*> deduplicatedEntryPoints;
    if (isRayTracingProgram)
    {
        std::set<std::string> entryPointNames;
        for (auto entryPoint : pTypeConformanceSpecializedEntryPoints)
        {
            auto compiledEntryPointName = std::string(entryPoint->getLayout()->getEntryPointByIndex(0)->getNameOverride());
            if (entryPointNames.find(compiledEntryPointName) == entryPointNames.end())
            {
                entryPointNames.insert(compiledEntryPointName);
                deduplicatedEntryPoints.push_back(entryPoint);
            }
        }
        programDesc.entryPointCount = (uint32_t)deduplicatedEntryPoints.size();
        programDesc.slangEntryPoints = (slang::IComponentType**)deduplicatedEntryPoints.data();
    }
    else
    {
        programDesc.entryPointCount = (uint32_t)pTypeConformanceSpecializedEntryPoints.size();
        programDesc.slangEntryPoints = (slang::IComponentType**)pTypeConformanceSpecializedEntryPoints.data();
    }

    Slang::ComPtr<ISlangBlob> diagnostics;
    if (SLANG_FAILED(pDevice->getGfxDevice()->createProgram(programDesc, pProgram->mGfxProgram.writeRef(), diagnostics.writeRef())))
    {
        pProgram = nullptr;
    }
    if (diagnostics)
    {
        log = (const char*)diagnostics->getBufferPointer();
    }

    return pProgram;
}

const EntryPointKernel* ProgramKernels::getKernel(ShaderType type) const
{
    for (auto& pEntryPointGroup : mUniqueEntryPointGroups)
    {
        if (auto pShader = pEntryPointGroup->getKernel(type))
            return pShader;
    }
    return nullptr;
}

ProgramVersion::ProgramVersion(Program* pProgram, slang::IComponentType* pSlangGlobalScope)
    : mpProgram(pProgram), mpSlangGlobalScope(pSlangGlobalScope)
{
    ASSERT(pProgram);
}

void ProgramVersion::init(
    const DefineList& defineList,
    const ref<const ProgramReflection>& pReflector,
    const std::string& name,
    const std::vector<Slang::ComPtr<slang::IComponentType>>& pSlangEntryPoints
)
{
    ASSERT(pReflector);
    mDefines = defineList;
    mpReflector = pReflector;
    mName = name;
    mpSlangEntryPoints = pSlangEntryPoints;
}

ref<ProgramVersion> ProgramVersion::createEmpty(Program* pProgram, slang::IComponentType* pSlangGlobalScope)
{
    return ref<ProgramVersion>(new ProgramVersion(pProgram, pSlangGlobalScope));
}

// ref<const ProgramKernels> ProgramVersion::getKernels(Device* pDevice, ProgramVars const* pVars) const
// {
//     // We need are going to look up or create specialized kernels
//     // based on how parameters are bound in `pVars`.
//     //
//     // To do this we need to identify those parameters that are relevant
//     // to specialization, and what argument type/value is bound to
//     // those parameters.
//     //
//     std::string specializationKey;
//
//     ParameterBlock::SpecializationArgs specializationArgs;
//     if (pVars)
//     {
//         pVars->collectSpecializationArgs(specializationArgs);
//     }
//
//     bool first = true;
//     for (auto specializationArg : specializationArgs)
//     {
//         if (!first)
//             specializationKey += ",";
//         specializationKey += std::string(specializationArg.type->getName());
//         first = false;
//     }
//
//     auto foundKernels = mpKernels.find(specializationKey);
//     if (foundKernels != mpKernels.end())
//     {
//         return foundKernels->second;
//     }
//
//     assert(!mpProgram);
//
//     // Loop so that user can trigger recompilation on error
//     for (;;)
//     {
//         std::string log;
//         auto pKernels = pDevice->getProgramManager()->createProgramKernels(*mpProgram, *this, *pVars, log);
//         if (pKernels)
//         {
//             // Success
//
//             if (!log.empty())
//             {
//                 printf("Warnings in program:\n%s\n%s", getName().c_str(), log.c_str());
//             }
//
//             mpKernels[specializationKey] = pKernels;
//             return pKernels;
//         }
//         else
//         {
//             // Failure
//             std::string msg = std::string("Failed to link program:\n") + getName() + std::string("\n")+log;
//             printf("%s\n", msg.c_str());
//             assert(0);
//         }
//     }
// }

slang::ISession* ProgramVersion::getSlangSession() const
{
    return getSlangGlobalScope()->getSession();
}

slang::IComponentType* ProgramVersion::getSlangGlobalScope() const
{
    return mpSlangGlobalScope;
}

slang::IComponentType* ProgramVersion::getSlangEntryPoint(uint32_t index) const
{
    return mpSlangEntryPoints[index];
}
