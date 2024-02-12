#include "DeviceWrapper.h"


Device::Device()
{
    printf("slang: create global session\n");
    slang::createGlobalSession(m_slangGlobalSession.writeRef());
    m_pProgramManager = std::make_unique<ProgramManager>(this);

    gfx::IDevice::Desc gfxDesc = {};
    gfxDesc.deviceType = gfx::DeviceType::Vulkan;
    gfxDesc.slang.slangGlobalSession = m_slangGlobalSession;
    gfxDesc.shaderCache.maxEntryCount = 1000;
    gfxDesc.shaderCache.shaderCachePath = nullptr;

    printf("gfx:: get GPU adapters\n");
    gfx::AdapterList adapters = gfx::gfxGetAdapters(gfxDesc.deviceType);
    if (adapters.getCount() == 0)
    {
        assert(!"No GPU found");
    }

    // Try to create device on specific GPU.
    gfxDesc.adapterLUID = &adapters.getAdapters()[0].luid;

    mpAPIDispatcher.reset(new PipelineCreationAPIDispatcher());
    gfxDesc.apiCommandDispatcher = static_cast<ISlangUnknown*>(mpAPIDispatcher.get());

    printf("gfx create device\n");
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

Device::~Device()
{
    m_pProgramManager.reset();
    m_gfxDevice.setNull();
    m_transientResourceHeaps.setNull();
    mpAPIDispatcher.reset();
}
