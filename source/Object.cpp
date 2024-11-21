/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include <set>
#include <mutex>
#include <cassert>
#include "Object.h"

#if FALCOR_ENABLE_OBJECT_TRACKING
static std::mutex sTrackedObjectsMutex;
static std::set<const Object*> sTrackedObjects;
#endif

void Object::incRef() const
{
    uint32_t refCount = mRefCount.fetch_add(1);
#if FALCOR_ENABLE_OBJECT_TRACKING
    if (refCount == 0)
    {
        std::lock_guard<std::mutex> lock(sTrackedObjectsMutex);
        sTrackedObjects.insert(this);
    }
#endif
}

void Object::decRef(bool dealloc) const noexcept
{
    uint32_t refCount = mRefCount.fetch_sub(1);
    if (refCount <= 0)
    {
        assert(!"Internal error: Object reference count < 0!");
    }
    else if (refCount == 1)
    {
#if FALCOR_ENABLE_OBJECT_TRACKING
        {
            std::lock_guard<std::mutex> lock(sTrackedObjectsMutex);
            sTrackedObjects.erase(this);
        }
#endif
        if (dealloc)
            delete this;
    }
}

#if FALCOR_ENABLE_OBJECT_TRACKING

void Object::dumpAliveObjects()
{
    std::lock_guard<std::mutex> lock(sTrackedObjectsMutex);
    logInfo("Alive objects:");
    for (const Object* object : sTrackedObjects)
        object->dumpRefs();
}

void Object::dumpRefs() const
{
    logInfo("Object (class={} address={}) has {} reference(s)", getClassName(), fmt::ptr(this), refCount());
#if FALCOR_ENABLE_REF_TRACKING
    std::lock_guard<std::mutex> lock(mRefTrackerMutex);
    for (const auto& it : mRefTrackers)
    {
        logInfo("ref={} count={}\n{}\n", it.first, it.second.count, it.second.origin);
    }
#endif
}

#endif // FALCOR_ENABLE_OBJECT_TRACKING

#if FALCOR_ENABLE_REF_TRACKING

void Object::incRef(uint64_t refId) const
{
    if (mEnableRefTracking)
    {
        std::lock_guard<std::mutex> lock(mRefTrackerMutex);
        auto it = mRefTrackers.find(refId);
        if (it != mRefTrackers.end())
        {
            it->second.count++;
        }
        else
        {
            mRefTrackers.emplace(refId, getStackTrace());
        }
    }

    incRef();
}

void Object::decRef(uint64_t refId, bool dealloc) const noexcept
{
    if (mEnableRefTracking)
    {
        std::lock_guard<std::mutex> lock(mRefTrackerMutex);
        auto it = mRefTrackers.find(refId);
        FALCOR_ASSERT(it != mRefTrackers.end());
        if (--it->second.count == 0)
        {
            mRefTrackers.erase(it);
        }
    }

    decRef();
}

void Object::setEnableRefTracking(bool enable)
{
    mEnableRefTracking = enable;
}

#endif // FALCOR_ENABLE_REF_TRACKING
