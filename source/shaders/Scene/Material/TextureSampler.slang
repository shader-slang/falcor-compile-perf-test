/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Interface for texture sampling techniques.
    Types implementing this interface support sampling using different LOD computation techniques
*/
[anyValueSize(32)]
interface ITextureSampler
{
    /** Sample from a 2D texture using the level of detail computed by this method
    */
    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv);
};

/** Texture sampling using implicit gradients from finite differences within quads.

    Shader model 6.5 and lower *only* supports gradient operations in pixel shaders.
    Usage in other shader stages will generate a compiler error.

    Shader model 6.6 adds support for gradient operations in other shader stages,
    but the application is responsible for making sure the neighboring threads
    within 2x2 quads have appropriate data for gradient computations to be valid.
*/
struct ImplicitLodTextureSampler : ITextureSampler
{
    __init()
    {
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.Sample(s, uv);
    }
};

/** Texture sampling using an explicit scalar level of detail.
*/
struct ExplicitLodTextureSampler : ITextureSampler
{
    float lod; ///< The explicit level of detail to use

    __init(float lod)
    {
        this.lod = lod;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.SampleLevel(s, uv, lod);
    }
};

/** Texture sampling using an explicit scalar level of detail using ray cones (with texture dimensions
    "subtracted" from the LOD value, and added back in before SampleLevel()).
*/
struct ExplicitRayConesLodTextureSampler : ITextureSampler
{
    float rayconesLODWithoutTexDims;    ///< this is \Delta_t, which is texture independent, plus the rest of the terms, except the texture size, which is added below

    __init(float rayconesLODWithoutTexDims)
    {
        this.rayconesLODWithoutTexDims = rayconesLODWithoutTexDims;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint txw, txh;
        t.GetDimensions(txw, txh);
        float lambda = 0.5 * log2(txw * txh) + rayconesLODWithoutTexDims;
        return t.SampleLevel(s, uv, lambda);
    }
};


/** Texture sampling using an explicit scalar, i.e., isotropic, level of detail using ray diffs,
    with the final LOD computations done below, since they are dependent on texture dimensions.
    Use ExplicitGradientTextureSampler if you want anisotropic filtering with ray diffs.
*/
struct ExplicitRayDiffsIsotropicTextureSampler : ITextureSampler
{
    enum class Mode { IsotropicOpenGLStyle, IsotropicPBRTStyle };
    static const Mode kMode = Mode::IsotropicOpenGLStyle;

    float2 dUVdx;               ///< derivatives in x over uv
    float2 dUVdy;               ///< derivatives in y over uv

    __init(float2 dUVdx, float2 dUVdy)
    {
        this.dUVdx = dUVdx;
        this.dUVdy = dUVdy;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint2 dim;
        t.GetDimensions(dim.x, dim.y);

        switch (kMode)
        {
        case Mode::IsotropicOpenGLStyle:
            {
                // Sharper, but alias sometimes for sharp edges textures.
                const float2 duvdx = dUVdx * dim.x;
                const float2 duvdy = dUVdy * dim.y;
                const float lambda = 0.5f * log2(max(dot(duvdx, duvdx), dot(duvdy, duvdy)));
                return t.SampleLevel(s, uv, lambda);
            }
        case Mode::IsotropicPBRTStyle:
            {
                // PBRT style (much blurrier, but never (?) aliases).
                const float filterWidth = 2.f * max(dim.x * max(abs(dUVdx.x), abs(dUVdy.x)), dim.y * max(abs(dUVdx.y), abs(dUVdy.y)));
                const float lambda = log2(filterWidth);
                return t.SampleLevel(s, uv, lambda);
            }
        }

        return float4(0.f);
    }
};

/** Texture sampling using explicit screen-space gradients.
*/
struct ExplicitGradientTextureSampler : ITextureSampler
{
    float2 gradX; ///< Gradient of texture coordinate in the screen-space X direction
    float2 gradY; ///< Gradient of texture coordiante in teh screen-space Y direction

    __init(float2 gradX, float2 gradY)
    {
        this.gradX = gradX;
        this.gradY = gradY;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.SampleGrad(s, uv, gradX, gradY);
    }
};

/** Texture sampling using filtered importance sampling
    The filters applied are controlled by the UV and LOD jitter values passed at init time.
*/
struct StochasticTextureSampler : ITextureSampler
{
    float2 gradX;           ///< Gradient of texture coordinate in the screen-space X direction
    float2 gradY;           ///< Gradient of texture coordiante in teh screen-space Y direction
    float2 uvJitter;        ///< Stochastic uv jitter
    float lodJitter;        ///< Stochastic lod jitter
    float maxAnisotropy;    ///< Max anisotropy (ratio between major and minor ellipse axes)

    __init(
    float2 gradX, float2 gradY, float2 uvJitter, float lodJitter, float maxAnisotropy=64)    
    {
        this.gradX = gradX;
        this.gradY = gradY;
        this.uvJitter = uvJitter;
        this.lodJitter = lodJitter;
        this.maxAnisotropy = maxAnisotropy;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {       
        uint2 dim;
        t.GetDimensions(dim.x, dim.y);    
        let dudx = dim.x * gradX.x;
        let dvdx = dim.y * gradX.y;
        let dudy = dim.x * gradY.x;
        let dvdy = dim.y * gradY.y;

        // Find min and max ellipse axis
        float2 maxAxis = float2(dudy, dvdy);
        float2 minAxis = float2(dudx, dvdx);
        if (dot(minAxis, minAxis) > dot(maxAxis, maxAxis))
        {
            minAxis = float2(dudy, dvdy);
            maxAxis = float2(dudx, dvdx);
        }

        float minAxisLength = length(minAxis);
        float maxAxisLength = length(maxAxis);
        
        if (minAxisLength > 0 && (minAxisLength * maxAnisotropy) < maxAxisLength)
        {
            let scale = maxAxisLength / (minAxisLength * maxAnisotropy);
            minAxisLength *= scale;
        }       

        // No need to clamp to min and max lod levels, as HW should do it for free.
        let lod = log2(minAxisLength) + lodJitter;

        uv = uv * dim + uvJitter;
        uv = (floor(uv) + 0.5) / dim;        

        return t.SampleLevel(s, uv, lod);
    }
};
