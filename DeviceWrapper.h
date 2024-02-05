#pragma once

#include <slang-com-ptr.h>
#include <slang-gfx.h>
#include <memory>
#include "Types.h"
#include "Object.h"
#include "ProgramManager.h"
class ProgramManager;

class GFXDebugCallBack : public gfx::IDebugCallback
{
    virtual void handleMessage(gfx::DebugMessageType type, gfx::DebugMessageSource source, const char* message) override
    {
        if (type == gfx::DebugMessageType::Error)
        {
            printf("GFX Error: %s", message);
        }
        else if (type == gfx::DebugMessageType::Warning)
        {
            printf("GFX Warning: %s", message);
        }
        else
        {
            printf("GFX Info: %s", message);
        }
    }
};

static GFXDebugCallBack gGFXDebugCallBack; // TODO: REMOVEGLOBAL

class Device  : public Object{
public:
    enum Type
    {
        Default, ///< Default device type, favors D3D12 over Vulkan.
        D3D12,
        Vulkan,
    };

    Device()
    {
        slang::createGlobalSession(m_slangGlobalSession.writeRef());
        m_pProgramManager = std::make_unique<ProgramManager>(this);

        gfx::IDevice::Desc gfxDesc = {};
        gfxDesc.deviceType = gfx::DeviceType::Vulkan;
        gfxDesc.slang.slangGlobalSession = m_slangGlobalSession;
        gfxDesc.shaderCache.maxEntryCount = 1000;
        gfxDesc.shaderCache.shaderCachePath = "";

        std::vector<void*> extendedDescs;
        // Add extended desc for root parameter attribute.
        gfx::D3D12DeviceExtendedDesc extDesc = {};
        extDesc.rootParameterShaderAttributeName = "root";
        extendedDescs.push_back(&extDesc);

        gfxDesc.extendedDescCount = extendedDescs.size();
        gfxDesc.extendedDescs = extendedDescs.data();

        gfx::AdapterList adapters = gfx::gfxGetAdapters(gfxDesc.deviceType);
        if (adapters.getCount() == 0)
        {
            assert(!"No GPU found");
        }

        // Try to create device on specific GPU.
        gfxDesc.adapterLUID = &adapters.getAdapters()[0].luid;
        if (SLANG_FAILED(gfx::gfxCreateDevice(&gfxDesc, m_gfxDevice.writeRef())))
        {
            printf("Failed to create device on GPU 0 (%s).", adapters.getAdapters()[0].name);
        }

        if (SLANG_FAILED(gfx::gfxSetDebugCallback(&gGFXDebugCallBack)))
        {
            printf("Failed to setup debug callback\n");
        }
        else
        {
            gfx::gfxEnableDebugLayer();
        }

        // Otherwise try create device on any available GPU.
        if (!m_gfxDevice)
        {
            gfxDesc.adapterLUID = nullptr;
            if (SLANG_FAILED(gfx::gfxCreateDevice(&gfxDesc, m_gfxDevice.writeRef())))
                assert(!"Failed to create device");
        }

        gfx::ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.flags = gfx::ITransientResourceHeap::Flags::AllowResizing;
        transientHeapDesc.constantBufferSize = 16 * 1024 * 1024;
        transientHeapDesc.samplerDescriptorCount = 2048;
        transientHeapDesc.uavDescriptorCount = 1000000;
        transientHeapDesc.srvDescriptorCount = 1000000;
        transientHeapDesc.constantBufferDescriptorCount = 1000000;
        transientHeapDesc.accelerationStructureDescriptorCount = 1000000;
        if (SLANG_FAILED(m_gfxDevice->createTransientResourceHeap(transientHeapDesc, m_transientResourceHeaps.writeRef()))) {
            assert(!"Fail to create transient source heaps");
        }
    }

    gfx::ITransientResourceHeap* getCurrentTransientResourceHeap()
    {
        return m_transientResourceHeaps.get();
    }

    ShaderModel getDefaultShaderModel() const
    {
        return ShaderModel::SM6_6;
    }
    ProgramManager* getProgramManager() const { return m_pProgramManager.get(); }

    slang::IGlobalSession* getSlangGlobalSession() const { return m_slangGlobalSession; }
    gfx::IDevice* getGfxDevice() const { return m_gfxDevice; }
    Type getType() const { return m_type; }

private:
    Slang::ComPtr<slang::IGlobalSession> m_slangGlobalSession;
    Slang::ComPtr<gfx::IDevice> m_gfxDevice;
    Slang::ComPtr<gfx::ITransientResourceHeap> m_transientResourceHeaps;
    Type m_type {Vulkan};
    std::unique_ptr<ProgramManager> m_pProgramManager;
};
