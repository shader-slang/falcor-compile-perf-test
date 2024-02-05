void mx_layer_bsdf(BSDF in1, BSDF in2, out BSDF result)
{
    // The inputs have been handled by the SlangShaderGenerator node walking preprocess
    result.setLayer(mxStack_nextLayer++);
    mxStack.layerWeights[result.getIndex()] = float3(1.f);
}

void mx_weighted_layer_bsdf(BSDF in1, float3 topWeight, BSDF in2, out BSDF result)
{
    // The inputs have been handled by the SlangShaderGenerator node walking preprocess
    result.setLayer(mxStack_nextLayer++);
    mxStack.layerWeights[result.getIndex()] = float3(1.f);

    if (in1.isBsdf())
        mxStack.bsdfWeights[in1.getIndex()] *= topWeight;
    else
        mxStack.layerWeights[in1.getIndex()] *= topWeight;
}
