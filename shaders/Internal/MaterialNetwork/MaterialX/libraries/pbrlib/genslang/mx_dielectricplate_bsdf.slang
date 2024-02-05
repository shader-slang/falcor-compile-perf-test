void mx_dielectricplate_bsdf(float weight, float3 tint, float ior, float2 roughness, float3 N, float3 U, int distribution, out BSDF bsdf)
{
    bsdf.setBsdf(mxStack_nextBsdf++);
    mxStack.bsdfWeights[bsdf.getIndex()] = float3(weight);

    mxStack.setBsdf(bsdf.getIndex(), createMxDielectricPlate(tint, ior, roughness, N, U, distribution));

    mxStack.bsdfN[bsdf.getIndex()] = normalize(N);
    mxStack.bsdfT[bsdf.getIndex()] = U;
    orthogonalizeVectors(mxStack.bsdfN[bsdf.getIndex()], mxStack.bsdfT[bsdf.getIndex()]);
}
