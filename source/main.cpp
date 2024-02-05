#include <stdio.h>
#include <slang-gfx.h>
#include <slang-com-ptr.h>
#include "Program.h"
#include "path-tracer.h"
#include "ProgramManager.h"
#include "DeviceWrapper.h"
#include "CpuTimer.h"
#include "Utility.h"

// Hard-code the type conformance list. The list is dumped from Falcor
// InternalPathTracerMaterial test.
void InitTypeConformanceList(TypeConformanceList& typeConformances)
{
    typeConformances.add("NullPhaseFunction", "IPhaseFunction", 0);
    typeConformances.add("IsotropicPhaseFunction", "IPhaseFunction", 1);
    typeConformances.add("HenyeyGreensteinPhaseFunction", "IPhaseFunction", 2);
    typeConformances.add("DualHenyeyGreensteinPhaseFunction", "IPhaseFunction", 3);

    typeConformances.add("StandardMaterial", "IMaterial", 1);
    typeConformances.add("ClothMaterial", "IMaterial", 2);
    typeConformances.add("HairMaterial", "IMaterial", 3);
    typeConformances.add("MERLMaterial", "IMaterial", 4);
    typeConformances.add("MERLMixMaterial", "IMaterial", 5);
    typeConformances.add("PBRTDiffuseMaterial", "IMaterial", 6);
    typeConformances.add("PBRTDiffuseTransmissionMaterial", "IMaterial", 7);
    typeConformances.add("PBRTConductorMaterial", "IMaterial", 8);
    typeConformances.add("PBRTDielectricMaterial", "IMaterial", 9);
    typeConformances.add("PBRTCoatedConductorMaterial", "IMaterial", 10);
    typeConformances.add("PBRTCoatedDiffuseMaterial", "IMaterial", 11);
    typeConformances.add("Layered_mixedLobes_Material", "IMaterial", 15);
}

void LoadShaderModules(ProgramDesc& desc)
{
    ProgramDesc::ShaderModuleList shaderModules;

    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/StandardMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("MaterialXTemp/MxLayered_mixedLobes_Material.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/ClothMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/HairMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/MERLMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/MERLMixMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/PBRT/PBRTCoatedConductorMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/PBRT/PBRTCoatedDiffuseMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/PBRT/PBRTConductorMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/PBRT/PBRTDiffuseMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/PBRT/PBRTDielectricMaterial.slang"));
    shaderModules.push_back(ProgramDesc::ShaderModule::fromFile("Rendering/Materials/PBRT/PBRTDiffuseTransmissionMaterial.slang"));

    desc.addShaderModules(shaderModules);
    desc.addShaderLibrary("RenderPasses/InternalPathTracer/TracePass.cs.slang").csEntry("main");
}

void TestCase(ref<Device>& device)
{
    TypeConformanceList typeConformances {};
    InitTypeConformanceList(typeConformances);

    PathTracer pathTracer {};
    DefineList defines = pathTracer.m_staticParams.getDefines(pathTracer);

    ProgramDesc desc;
    LoadShaderModules(desc);

    desc.addTypeConformances(typeConformances);

    ref<Program> pProg = Program::create(device, desc, defines);

    for (uint32_t i = 0; i < 2; i++)
    {
        // Each set of pair of `Macro defines` and `Type conformance object` can define
        // one version of program.
        printf("Start creating program versions\n");
        const ref<const ProgramVersion>& progVersion = pProg->getActiveVersion();

        std::string log;
        ref<const ProgramKernels> programKernel = device->getProgramManager()->createProgramKernels(*pProg, *progVersion, log);
        const EntryPointKernel* entryPointKernel = programKernel->getKernel(ShaderType::Compute);

        Slang::ComPtr<gfx::IShaderObject> shaderObject;
        SlangResult res = device->getGfxDevice()->createMutableRootShaderObject(programKernel->getGfxProgram(), shaderObject.writeRef());
        ASSERT_EQ(res, SLANG_OK, "createMutableRootShaderObject");

        Slang::ComPtr<gfx::ICommandBuffer> gfxCommandBuffer;
        res = device->getCurrentTransientResourceHeap()->createCommandBuffer(gfxCommandBuffer.writeRef());
        ASSERT_EQ(res, SLANG_OK, "createCommandBuffer");

        gfx::IComputeCommandEncoder* computeCommandEncoder = gfxCommandBuffer->encodeComputeCommands();
        ASSERT_EQ(res, SLANG_OK, "encodeComputeCommands");

        // Create a pipeline state with the loaded shader.
        gfx::ComputePipelineStateDesc computePipelineDesc = {};
        Slang::ComPtr<gfx::IPipelineState> gfxPipelineState {nullptr};
        computePipelineDesc.program = programKernel->getGfxProgram();
        res = device->getGfxDevice()->createComputePipelineState(computePipelineDesc, gfxPipelineState.writeRef());
        ASSERT_EQ(res, SLANG_OK, "createComputePipelineState");

        res = computeCommandEncoder->bindPipelineWithRootObject(gfxPipelineState, shaderObject);
        ASSERT_EQ(res, SLANG_OK, "bindPipelineWithRootObject");

        CpuTimer timer;
        timer.update();

        res = computeCommandEncoder->dispatchCompute(0, 0, 0);
        ASSERT_EQ(res, SLANG_OK, "create pipeline");

        timer.update();
        double time = timer.delta();
        if (i == 0)
        {
            printf("Time for compiling spirv generated by glslang: %.3fs\n\n\n", time);
        }
        else
        {
            printf("Time for compiling spirv generated by slang: %.3fs\n\n\n", time);
        }

        device->getProgramManager()->reloadAllPrograms();
        device->getProgramManager()->setSpirvDirectMode(true);
    }
}

int main(int argc, char* argv[])
{
    ref<Device> device = make_ref<Device>();
    TestCase(device);

    return 0;
}
