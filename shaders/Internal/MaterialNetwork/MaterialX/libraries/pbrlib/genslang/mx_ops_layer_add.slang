void mx_add_bsdf(BSDF in1, BSDF in2, out BSDF result)
{
    // The inputs have been handled by the SlangShaderGenerator node walking preprocess
    result.setLayer(mxStack_nextLayer++);
    mxStack.layerWeights[result.getIndex()] = float3(1.f);
}
