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
#pragma once
#include <slang.h>

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "Object.h"
#include "Types.h"

class ProgramVersion;
class ReflectionVar;
class ReflectionType;
class ReflectionResourceType;
class ReflectionBasicType;
class ReflectionStructType;
class ReflectionArrayType;
class ReflectionInterfaceType;
class ParameterBlockReflection;

class  ReflectionType : public Object
{
    FALCOR_OBJECT(ReflectionType)
public:
    virtual ~ReflectionType() = default;

    /**
     * The kind of a type.
     *
     * Every type has a kind, which specifies which subclass of `ReflectionType` it uses.
     *
     * When adding new derived classes, this enumeration should be updated.
     */
    enum class Kind
    {
        Array,     ///< ReflectionArrayType
        Struct,    ///< ReflectionStructType
        Basic,     ///< ReflectionBasicType
        Resource,  ///< ReflectionResourceType
        Interface, ///< ReflectionInterfaceType
    };
};

class EntryPointGroupReflection : public Object
{
public:
    static ref<EntryPointGroupReflection> create(
        ProgramVersion const* pProgramVersion,
        uint32_t groupIndex,
        const std::vector<slang::EntryPointLayout*>& pSlangEntryPointReflectors
    );

private:
    EntryPointGroupReflection(ProgramVersion const* pProgramVersion) { };
};

typedef EntryPointGroupReflection EntryPointBaseReflection;

/**
 * Reflection object for an entire program. Essentially, it's a collection of ParameterBlocks
 */
class ProgramReflection : public Object
{
public:

    /**
     * Create a new object for a Slang reflector object
     */
    static ref<const ProgramReflection> create(
        ProgramVersion const* pProgramVersion,
        slang::ShaderReflection* pSlangReflector,
        const std::vector<slang::EntryPointLayout*>& pSlangEntryPointReflectors,
        std::string& log
    );

    void finalize();

    ProgramVersion const* getProgramVersion() const { return mpProgramVersion; }
    const ref<EntryPointGroupReflection>& getEntryPointGroup(uint32_t index) const { return mEntryPointGroups[index]; }

private:
    ProgramReflection(
        ProgramVersion const* pProgramVersion,
        slang::ShaderReflection* pSlangReflector,
        const std::vector<slang::EntryPointLayout*>& pSlangEntryPointReflectors,
        std::string& log
    );
    ProgramReflection(ProgramVersion const* pProgramVersion) {}
    ProgramReflection(const ProgramReflection&) = default;

    ProgramVersion const* mpProgramVersion;

    slang::ShaderReflection* mpSlangReflector = nullptr;

    std::vector<ref<EntryPointGroupReflection>> mEntryPointGroups;
};
