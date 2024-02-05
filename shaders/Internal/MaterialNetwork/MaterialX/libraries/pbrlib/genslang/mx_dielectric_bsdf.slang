void mx_dielectric_bsdf(
    float weight,
    float3 tint,
    float ior,
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

    mxStack.setBsdf(bsdf.getIndex(), createMxDielectric(tint, ior, roughness, N, U, distribution, scatter_mode, sd.frontFacing));

    mxStack.bsdfN[bsdf.getIndex()] = normalize(N);
    mxStack.bsdfT[bsdf.getIndex()] = U;
    orthogonalizeVectors(mxStack.bsdfN[bsdf.getIndex()], mxStack.bsdfT[bsdf.getIndex()]);
}
