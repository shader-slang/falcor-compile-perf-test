void mx_oren_nayar_diffuse_bsdf(float weight, float3 color, float roughness, float3 N, out BSDF bsdf)
{
    bsdf.setBsdf(mxStack_nextBsdf++);
    mxStack.bsdfWeights[bsdf.getIndex()] = float3(weight);

    mxStack.setBsdf(bsdf.getIndex(), createMxOrenNayar(color, roughness, N));

    mxStack.bsdfN[bsdf.getIndex()] = normalize(N);
    mxStack.bsdfT[bsdf.getIndex()] = sd.frame.T;
    orthogonalizeVectors(mxStack.bsdfN[bsdf.getIndex()], mxStack.bsdfT[bsdf.getIndex()]);
}
