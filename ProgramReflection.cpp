/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
