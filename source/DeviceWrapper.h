#pragma once

#if defined(Linux)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#error "No OS specified"
#endif

#include <slang-com-ptr.h>
#include <slang-gfx.h>
#include <memory>
#include "Types.h"
#include "Object.h"
#include "ProgramManager.h"
#include <vulkan/vulkan.h>
#include "CpuTimer.h"

class ProgramManager;
class PipelineCreationAPIDispatcher;

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

class PipelineCreationAPIDispatcher : public gfx::IPipelineCreationAPIDispatcher
{
public:
    PipelineCreationAPIDispatcher() { }
    ~PipelineCreationAPIDispatcher() { }

    double getPipelineCreationTime() {return m_timer.delta();}

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject) override
    {
        if (uuid == SlangUUID SLANG_UUID_IVulkanPipelineCreationAPIDispatcher)
        {
            *outObject = static_cast<gfx::IPipelineCreationAPIDispatcher*>(this);
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }

    // The lifetime of this dispatcher object will be managed by `Falcor::Device` so we don't need
    // to actually implement reference counting here.
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 2; }

    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 2; }

    // This method will be called by the gfx layer to create an API object for a compute pipeline state.
    virtual gfx::Result createComputePipelineState(
        gfx::IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState
    )
    {
        const char* dynamicLibraryName = "Unknown";

#if defined(Linux)
        dynamicLibraryName = "libvulkan.so.1";
        void* vulkanLibraryHandle = dlopen(dynamicLibraryName, RTLD_NOW);
#elif defined(_WIN32)
        dynamicLibraryName = "vulkan-1.dll";
        HMODULE vulkanLibraryHandle = ::LoadLibraryA(dynamicLibraryName);
#else
#error "No OS specified"
#endif

        gfx::IDevice::InteropHandles outHandles;
        device->getNativeDeviceHandles(&outHandles);

        VkInstance instance;
        instance = (VkInstance)outHandles.handles[0].handleValue;

        VkDevice vkDevice;
        vkDevice = (VkDevice)outHandles.handles[2].handleValue;

        PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;

#if defined(Linux)
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(vulkanLibraryHandle, "vkGetInstanceProcAddr");
#elif defined(_WIN32)
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)::GetProcAddress(vulkanLibraryHandle, "vkGetInstanceProcAddr");
#else
#error "No OS specified"
#endif

        if (!vkGetInstanceProcAddr)
        {
            assert(!"Fail to get instance proc address");
        }

        PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
        vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr");
        if (!vkGetDeviceProcAddr)
        {
            assert(!"Fail to get device proc address");
        }

        PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
        vkCreateComputePipelines = (PFN_vkCreateComputePipelines)vkGetDeviceProcAddr(vkDevice, "vkCreateComputePipelines");
        if (!vkCreateComputePipelines)
        {
            assert(!"Fail to vkCreateComputePipelines");
        }

        m_timer.update();

        VkPipelineCache pipelineCache = VK_NULL_HANDLE;
        VkComputePipelineCreateInfo* pComputePipelineInfo = static_cast<VkComputePipelineCreateInfo*>(pipelineDesc);
        VkPipeline pipeline;
        vkCreateComputePipelines(
            vkDevice, pipelineCache, 1, pComputePipelineInfo, nullptr, &pipeline);

        *((VkPipeline*)outPipelineState) = pipeline;
        m_timer.update();
        return SLANG_OK;
    }

    // This method will be called by the gfx layer to create an API object for a graphics pipeline state.
    virtual gfx::Result createGraphicsPipelineState(
        gfx::IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState
    )
    {
        assert(!"Not implemented!");
        return SLANG_OK;
    }

    virtual gfx::Result createMeshPipelineState(
        gfx::IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState
    )
    {
        assert(!"Mesh pipelines are not supported.");
        return SLANG_OK;
    }

    // This method will be called by the gfx layer right before creating a ray tracing state object.
    virtual gfx::Result beforeCreateRayTracingState(gfx::IDevice* device, slang::IComponentType* program)
    {
        assert(!"Not implemented!");
        return SLANG_OK;
    }

    // This method will be called by the gfx layer right after creating a ray tracing state object.
    virtual gfx::Result afterCreateRayTracingState(gfx::IDevice* device, slang::IComponentType* program)
    {
        assert(!"Not implemented!");
        return SLANG_OK;
    }
private:
    CpuTimer m_timer;
};

class Device  : public Object{
public:
    enum Type
    {
        Default, ///< Default device type, favors D3D12 over Vulkan.
        D3D12,
        Vulkan,
    };
    Device();
    ~Device();

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

    double getPipelineCreationTime() {return mpAPIDispatcher->getPipelineCreationTime();}
private:
    Slang::ComPtr<slang::IGlobalSession> m_slangGlobalSession;
    Slang::ComPtr<gfx::IDevice> m_gfxDevice;
    Slang::ComPtr<gfx::ITransientResourceHeap> m_transientResourceHeaps;
    Type m_type {Vulkan};
    std::unique_ptr<ProgramManager> m_pProgramManager;
    std::unique_ptr<PipelineCreationAPIDispatcher> mpAPIDispatcher;
};
