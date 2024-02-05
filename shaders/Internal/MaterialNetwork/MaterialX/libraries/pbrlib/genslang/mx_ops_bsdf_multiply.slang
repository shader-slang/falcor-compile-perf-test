void mx_multiply_bsdf(BSDF in1, float3 in2, out BSDF result)
{
    if (in1.isBsdf())
        mxStack.bsdfWeights[in1.getIndex()] *= in2;
    else
        mxStack.layerWeights[in1.getIndex()] *= in2;
    result = in1;
}

void mx_multiply_bsdf(BSDF in1, float in2, out BSDF result)
{
    if (in1.isBsdf())
        mxStack.bsdfWeights[in1.getIndex()] *= in2;
    else
        mxStack.layerWeights[in1.getIndex()] *= in2;
    result = in1;
}
