void mx_roughness_dual(float2 roughness, out float2 result)
{
    result.x = clamp(roughness.x * roughness.x, M_FLOAT_EPS, 1.0);
    if (roughness.y < 0.0)
    {
        result.y = result.x;
    }
    else
    {
        result.y = clamp(roughness.y * roughness.y, M_FLOAT_EPS, 1.0);
    }
}
