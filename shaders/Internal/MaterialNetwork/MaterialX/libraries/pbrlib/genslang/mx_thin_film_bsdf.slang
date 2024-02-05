void mx_thin_film_bsdf(float thickness, float ior, out BSDF bsdf)
{
    bsdf.setBsdf(mxStack_nextBsdf++);
    mxStack.bsdfWeights[bsdf.getIndex()] = float3(weight);

    mxStack.setBsdf(bsdf.getIndex(), createMxThinFilm(thickness, ior));
    // Already orthonormalized, no need to call computeLobeTangentSpace
    mxStack.bsdfN[bsdf.getIndex()] = sd.frame.N;
    mxStack.bsdfT[bsdf.getIndex()] = sd.frame.T;
}
