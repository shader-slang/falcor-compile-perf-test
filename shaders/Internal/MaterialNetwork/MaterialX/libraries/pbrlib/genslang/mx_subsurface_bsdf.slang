void mx_subsurface_bsdf(float weight, float3 color, float3 radius, float anisotropy, float3 N, out BSDF bsdf)
{
    bsdf.setBsdf(mxStack_nextBsdf++);
    mxStack.bsdfWeights[bsdf.getIndex()] = float3(weight);

    mxStack.setBsdf(bsdf.getIndex(), createMxTranslucent(float3(1.f), N));

    mxStack.bsdfN[bsdf.getIndex()] = normalize(N);
    mxStack.bsdfT[bsdf.getIndex()] = sd.frame.T;
    orthogonalizeVectors(mxStack.bsdfN[bsdf.getIndex()], mxStack.bsdfT[bsdf.getIndex()]);

    mxStack.subsurfaceData.color = color;
    mxStack.subsurfaceData.radius = radius;
    mxStack.subsurfaceData.N = mxStack.bsdfN[bsdf.getIndex()];
    mxStack.subsurfaceData.T = mxStack.bsdfT[bsdf.getIndex()];
    mxStack.subsurfaceData.single_hg(anisotropy);
}
