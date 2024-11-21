/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Utilities for converting tristimulus values to/from spectral data.
 */
struct SpectrumUtils
{
    /**
     * Evaluates the 1931 CIE XYZ color matching curves.
     * This uses a multi-lobe piecewise Gaussian fit described in:
     * Wyman et al., "Simple Analytic Approximations to the CIE XYZ Color
     * Matching Functions", JCGT, vol 2(2), 2013.
     * @param[in] lambda Wavelength in nm.
     * @return XYZ tristimulus values.
     */
    static float3 wavelengthToXYZ(float lambda)
    {
        float x = 0.362f * G(lambda, 442.0f, 0.0624f, 0.0374f) + 1.056f * G(lambda, 599.8f, 0.0264f, 0.0323f) -
                  0.065f * G(lambda, 501.1f, 0.0490f, 0.0382f);
        float y = 0.821f * G(lambda, 568.8f, 0.0213f, 0.0247f) + 0.286f * G(lambda, 530.9f, 0.0613f, 0.0322f);
        float z = 1.217f * G(lambda, 437.0f, 0.0845f, 0.0278f) + 0.681f * G(lambda, 459.0f, 0.0385f, 0.0725f);
        return float3(x, y, z);
    }

    // Internal

    static float G(float x, float lambda, float invSigma1, float invSigma2)
    {
        float t = (x - lambda) * (x < lambda ? invSigma1 : invSigma2);
        return exp(-0.5f * t * t);
    }
};
