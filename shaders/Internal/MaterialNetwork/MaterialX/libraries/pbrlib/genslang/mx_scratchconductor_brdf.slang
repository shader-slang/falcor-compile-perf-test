void mx_scratchconductor_brdf(
    float weight,
    float3 ior_n,
    float3 ior_k,
    float2 roughness,
    float3 scratchdirection,
    float depth,
    float3 mask,
    float scratchRoughness,
    float3 N,
    float3 U,
    int distribution,
    out BSDF bsdf
)
{
    bsdf.setBsdf(mxStack_nextBsdf++);
    mxStack.bsdfWeights[bsdf.getIndex()] = float3(weight);
    mxStack.bsdfN[bsdf.getIndex()] = normalize(N);
    mxStack.bsdfT[bsdf.getIndex()] = sd.frame.T;
    orthogonalizeVectors(mxStack.bsdfN[bsdf.getIndex()], mxStack.bsdfT[bsdf.getIndex()]);

    mxStack.setBsdf(
        bsdf.getIndex(),
        createMxScratchConductor(
            ior_n,
            ior_k,
            roughness,
            float3(scratchdirection[1], scratchdirection[0], scratchdirection[2]) * 2.0f - 1.0f,
            max(mask[1] * depth, 0.0001f),
            float3(clamp(mask[0], 0.0f, 1.0f), mask[1], mask[2]),
            clamp(scratchRoughness, 0.f, 1.f),
            N,
            U,
            distribution
        )
    );
}
