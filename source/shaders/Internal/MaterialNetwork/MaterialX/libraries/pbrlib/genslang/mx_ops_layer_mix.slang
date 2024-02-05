void mx_mix_bsdf(BSDF in1, BSDF in2, float mix, out BSDF result)
{
    // The inputs have been handled by the SlangShaderGenerator node walking preprocess
    result.setLayer(mxStack_nextLayer++);
    mxStack.layerWeights[result.getIndex()] = float3(1.f);

    if (in1.isBsdf())
        mxStack.bsdfWeights[in1.getIndex()] *= mix;
    else
        mxStack.layerWeights[in1.getIndex()] *= mix;

    if (in2.isBsdf())
        mxStack.bsdfWeights[in2.getIndex()] *= (1 - mix);
    else
        mxStack.layerWeights[in2.getIndex()] *= (1 - mix);
}
