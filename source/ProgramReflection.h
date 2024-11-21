/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
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
