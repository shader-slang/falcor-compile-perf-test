void mx_generalized_schlick_bsdf(
    float weight,
    float3 color0,
    float3 color90,
    float exponent,
    float2 roughness,
    float3 N,
    float3 U,
    int distribution,
    int scatter_mode,
    out BSDF bsdf
)
{
    bsdf.setBsdf(mxStack_nextBsdf++);
    mxStack.bsdfWeights[bsdf.getIndex()] = float3(weight);

    mxStack.setBsdf(bsdf.getIndex(), createMxGeneralizedSchlick(color0, color90, exponent, roughness, N, U, distribution, scatter_mode));

    mxStack.bsdfN[bsdf.getIndex()] = normalize(N);
    mxStack.bsdfT[bsdf.getIndex()] = U;
    orthogonalizeVectors(mxStack.bsdfN[bsdf.getIndex()], mxStack.bsdfT[bsdf.getIndex()]);
}
