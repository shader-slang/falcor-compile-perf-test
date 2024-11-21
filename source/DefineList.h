/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once

#include <initializer_list>
#include <map>
#include <string>

class DefineList : public std::map<std::string, std::string>
{
public:
    /**
     * Adds a macro definition. If the macro already exists, it will be replaced.
     * @param[in] name The name of macro.
     * @param[in] value Optional. The value of the macro.
     * @return The updated list of macro definitions.
     */
    DefineList& add(const std::string& name, const std::string& val = "")
    {
        (*this)[name] = val;
        return *this;
    }

    /**
     * Removes a macro definition. If the macro doesn't exist, the call will be silently ignored.
     * @param[in] name The name of macro.
     * @return The updated list of macro definitions.
     */
    DefineList& remove(const std::string& name)
    {
        (*this).erase(name);
        return *this;
    }

    /**
     * Add a define list to the current list
     */
    DefineList& add(const DefineList& dl)
    {
        for (const auto& p : dl)
            add(p.first, p.second);
        return *this;
    }

    /**
     * Remove a define list from the current list
     */
    DefineList& remove(const DefineList& dl)
    {
        for (const auto& p : dl)
            remove(p.first);
        return *this;
    }

    DefineList() = default;
    DefineList(std::initializer_list<std::pair<const std::string, std::string>> il) : std::map<std::string, std::string>(il) {}
};
