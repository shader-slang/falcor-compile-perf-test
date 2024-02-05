/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Utils.PixelStats;
import GridVolumeSamplerParams;

#ifndef GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR
// GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR must be defined in order to use this module.
#error "GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR not defined!"
#endif

#ifndef GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER
// GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER must be defined in order to use this module.
#error "GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER not defined!"
#endif

#ifndef GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
// GRID_VOLUME_SAMPLER_USE_BRICKEDGRID must be defined in order to use this module.
#error "GRID_VOLUME_SAMPLER_USE_BRICKEDGRID not defined!"
#endif

/** Helper class for sampling grid volumes in the scene.
    Note: For simplicity, this sampler only uses the first volume in the scene.
*/
struct GridVolumeSampler
{
    static const TransmittanceEstimator kTransmittanceEstimator = TransmittanceEstimator(GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR);
    static const DistanceSampler kDistanceSampler = DistanceSampler(GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER);

    uint4 _dummy;

    // Multi-level DDA settings.
    static const float kDDAMipStart = 2.f;
    static const float kDDAMipUpSpeed = 0.33f;
    static const float kDDAMipDownSpeed = 2.f;

#ifdef FALCOR_INTERNAL
    // Ray marching settings.
    static const float kRayMarchingRouletteThreshold = GRID_VOLUME_SAMPLER_RAY_MARCHING_ROULETTE_THRESHOLD;
    static const float kBiasedRayMarchingLookupMultiplier = GRID_VOLUME_SAMPLER_BIASED_RAY_MARCHING_LOOKUP_MULTIPLIER;
    static const float kBiasedRayMarchingLocalCDFStepSize = GRID_VOLUME_SAMPLER_BIASED_RAY_MARCHING_LOCAL_CDF_STEP_SIZE;
    static const float kUnbiasedRayMarchingLookupMultiplier = GRID_VOLUME_SAMPLER_UNBIASED_RAY_MARCHING_LOOKUP_MULTIPLIER;
    static const uint kUnbiasedRayMarchingMaxOrder = GRID_VOLUME_SAMPLER_UNBIASED_RAY_MARCHING_MAX_ORDER;
    static const uint kUnbiasedRayMarchingMaxSamples = kUnbiasedRayMarchingMaxOrder + 1;

    /** Target density lookup count for Ray Marching based algorithms.

        The lookup count is based on the control optical thickness tau (e.g. the majorant integral) and follows
        An unbiased ray-marching transmittance estimator (Kettunen et al. 2021).

        \param[in] tau The control optical thickness, e.g. the majorant integral or majorant integral - minorant integral.
        \return The target density lookup count
    */
    static float evalRayMarchingTargetLookupCount(float tau)
    {
        // Match lookup count to p-series CMF.
        return ceil(pow((0.015f + tau) * (0.65f + tau) * (60.3f + tau), 1.f / 3.0f));
    }

    /** Expected sample count for the Bhanot & Kennedy roulette.
        See An unbiased ray-marching transmittance estimator (Kettunen et al. 2021).

        \param[in] C The B&K C parameter of the roulette (minimum evaluation order)
        \return Expected sample count
    */
    static float evalBhanotKennedySampleCount(float C)
    {
        // Return N + N! / C^N * [exp(C) - sum[k=0..N] c^k/k!]
        // where N = floor(C)
        int N = int(C);

        float t = 1.f;
        float taylorSum = 1.f;
        for (int k = 1; k <= N; ++k)
        {
            t *= C / (float)k;
            taylorSum += t;
        }

        return N + (exp(C) - taylorSum) / t;
    }

    /** Russian roulette for Unbiased Ray Marching.
        See An unbiased ray-marching transmittance estimator (Kettunen et al. 2021).
    */
    struct UnbiasedRayMarchingRoulette {
        static const float kBhanotKennedyC = 2.f; // Hard-coded for C=2.
        static const float kPZero = 0.9f;

        float u; ///< A random number that selects the evaluation order.
        float P; ///< Acceptance probability of the current order.

        __init(float random)
        {
            this.u = random;
            this.P = 1.f;
        }

        /** Evaluates the tuple size for the transmittance estimator.
            See An unbiased ray-marching transmittance estimator (Kettunen et al. 2021).

            \param[in] controlIntegral The control optical thickness, e.g. the majorant integral or majorant integral - minorant integral, possibly multiplied by a quality control parameter.
            \return Number of density lookups per sample.
        */
        static uint evalTupleSize(float controlIntegral)
        {
            float targetLookupCount = evalRayMarchingTargetLookupCount(controlIntegral);
            float sampleCount = 1.f + (1.f - kPZero) * evalBhanotKennedySampleCount(kBhanotKennedyC);
            return (uint)(max(1.f, 0.5f + targetLookupCount / sampleCount));
        }

        /** Rewinds the state for deterministic replay.
        */
        [mutating] void rewind()
        {
            P = 1.f;
        }

        /** Acceptance probability for order k, or 0 if roulette failed.
        */
        [mutating] float accept(uint k)
        {
            // Hardcoded for kBhanotKennedyC = 2.
            float P_accept = 1.f - kPZero;
            if (k > 1) P_accept = kBhanotKennedyC / (float)k;

            P *= P_accept;
            return (P > u) ? P_accept : 0.f;
        }
    };

    /** Evaluates the elementary symmetric means / U-statistics of the given samples.

        Subtracts the pivot from the samples before evaluation and skips the pivot from the means.
        See An unbiased ray-marching transmittance estimator (Kettunen et al. 2021).

        \param[in] samples The independent optical thickness samples
        \param[in] maxOrder Evaluation order of the power series
        \param[in] pivotIndex Index of the sample used as the pivot
        \param[out] m The elementary means of orders 0 to maxOrder.
        \return Elementary symmetric means of the shifted samples.
    */
    void evalShiftedElementaryMeans(const float samples[kUnbiasedRayMarchingMaxSamples], uint maxOrder, uint pivotIndex, out float m[kUnbiasedRayMarchingMaxOrder + 1])
    {
        // See An unbiased ray-marching transmittance estimator (Kettunen et al. 2021).
        const float pivot = samples[pivotIndex];

        m[0] = 1.f;
        for (uint i = 1; i <= maxOrder; i++)
        {
            m[i] = 0.f;
        }

        // The elementary symmetric means algorithm from the paper. The same loop twice but but skip the pivot index.
        for (uint n = 1; n <= pivotIndex; n++)
        {
            float x = samples[n - 1] - pivot;
            for (uint k = n; k >= 1; k--)
            {
                m[k] += (float(k) / (float)n) * (m[k - 1] * x - m[k]);
            }
        }

        for (uint n = pivotIndex + 1; n <= maxOrder; n++)
        {
            float x = samples[n] - pivot;
            for (uint k = n; k >= 1; k--)
            {
                m[k] += (float(k) / (float)n) * (m[k - 1] * x - m[k]);
            }
        }
    }
#endif

    /** Transform a ray from world space to grid index space.
        \param[in] gridVolume Grid volume object that stores the transform matrix we need.
        \param[in] densityGrid Grid object that stores the index space bounding box with which we need to offset coordinates, when using BrickedGrid.
        \param[in] rayOriginWorld Ray origin in world space.
        \param[in] rayDirWorld Ray direction in world space.
        \param[out] rayOriginIndex Ray origin in index space.
        \param[out] rayDirIndex Ray direction in index space (non-normalized!).
    */
    void transformRayToIndexSpace(const GridVolume gridVolume, const Grid densityGrid, const float3 rayOriginWorld, const float3 rayDirWorld, out float3 rayOriginIndex, out float3 rayDirIndex) {
        rayOriginIndex = mul(gridVolume.data.invTransform, float4(rayOriginWorld, 1.f)).xyz;
        rayDirIndex = mul(gridVolume.data.invTransform, float4(rayDirWorld, 0.f)).xyz;
#if GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
        // NanoVDB allows for negative indices, while textures require positive indices.
        // Thus, we simply translate the index-space box, matching a shift we do at grid conversion time on the host.
        rayOriginIndex -= densityGrid.getMinIndex();
#endif
    }

#if GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
    typedef float DensityAccessor; // dummy type to make interface to lookupDensityTex the same for both NanoVDB and Density Grid
    DensityAccessor createAccessor(const Grid densityGrid) { return kDDAMipStart; }
    float getAccessorLevel(const DensityAccessor a) { return a; }
    void increaseAccessorLevel(inout DensityAccessor a) { a = min(3.f, a + kDDAMipUpSpeed); }
    void decreaseAccessorLevel(inout DensityAccessor a) { a = max(0.f, a - kDDAMipDownSpeed); }
#else
    typedef Grid::Accessor DensityAccessor;
    DensityAccessor createAccessor(const Grid densityGrid) { return densityGrid.createAccessor(); }
    float getAccessorLevel(const DensityAccessor a) { return 0.f; }
    void increaseAccessorLevel(inout DensityAccessor a) { }
    void decreaseAccessorLevel(inout DensityAccessor a) { }
#endif

    struct DistanceSample
    {
        float t;    ///< Sampled distance.
        float3 thp; ///< Throughput.
    };

    /** Checks if a ray potentially intersects a volume.
        \param[in] ray Ray.
        \return Returns true if a volume potentially intersects the ray.
    */
    bool intersectsVolumes(const Ray ray)
    {
        if (gScene.getGridVolumeCount() == 0) return false;
        float2 nearFar;
        return intersectVolume(gScene.getGridVolume(0), ray, nearFar);
    }

    /** Perform a DDA step on a given mipmap level of (dilated) bricks.
        \param[in] rayOrigin Ray origin.
        \param[in] invRayDir Inverse ray direction.
        \param[in] pos Current position along ray.
        \param[in] mip Range texture mipmap level to traverse on (8^mip + 2 voxels).
        \return Distance from the ray origin to the next intersection with (dilated) brick borders.
    */
    float stepDDA(const float3 rayOrigin, const float3 invRayDir, const float3 pos, const int mip)
    {
        const float dim = 8 << mip;
        const float3 tofs = invRayDir * (select(invRayDir >= 0.f, dim + 0.5f, -0.5f) - rayOrigin);
        const float3 tmax = floor(pos * (1.f / dim)) * dim * invRayDir + tofs;
        return min(tmax.x, min(tmax.y, tmax.z));
    }

    /** Evaluate transmittance along a ray through volumes.
        \param[in] ray Ray.
        \param[in,out] sg Sample generator.
        \return Returns the transmittance.
    */
    float evalTransmittance<S : ISampleGenerator>(const Ray ray, inout S sg)
    {
        if (gScene.getGridVolumeCount() == 0) return 1.f;

        GridVolume gridVolume = gScene.getGridVolume(0);
        if (!gridVolume.hasDensityGrid()) return 1.f;

        float2 nearFar;
        if (!intersectVolume(gridVolume, ray, nearFar)) return 1.f;

        TinyUniformSampleGenerator tsg = TinyUniformSampleGenerator(jenkinsHash(sg.next()));

        switch (kTransmittanceEstimator)
        {
        case TransmittanceEstimator::DeltaTracking:
            return evalTransmittanceDeltaTracking(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        case TransmittanceEstimator::RatioTracking:
            return evalTransmittanceRatioTracking(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        case TransmittanceEstimator::RatioTrackingLocalMajorant:
            return evalTransmittanceRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, nearFar, tsg);
#ifdef FALCOR_INTERNAL
        case TransmittanceEstimator::UnbiasedRayMarching:
            return evalTransmittanceUnbiasedRayMarching(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        case TransmittanceEstimator::BiasedRayMarching:
            return evalTransmittanceBiasedRayMarching(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        case TransmittanceEstimator::BiasedRayMarchingLocalMajorant:
            return evalTransmittanceBiasedRayMarchingLocalDDA(gridVolume, ray.origin, ray.dir, nearFar, tsg);
#endif
        default:
            return 1.f;
        }
    }

    /** Sample a scattering distance along a ray through volumes.
        \param[in] ray Ray.
        \param[in,out] sg Sample generator.
        \param[out] ds Distance sample.
        \return Returns true if a valid scattering distance was sampled.
    */
    bool sampleDistance<S : ISampleGenerator>(const Ray ray, inout S sg, out DistanceSample ds)
    {
        ds = {};

        if (gScene.getGridVolumeCount() == 0) return false;

        GridVolume gridVolume = gScene.getGridVolume(0);
        if (!gridVolume.hasDensityGrid()) return false;

        float2 nearFar;
        if (!intersectVolume(gridVolume, ray, nearFar)) return false;

        TinyUniformSampleGenerator tsg = TinyUniformSampleGenerator(jenkinsHash(sg.next()));

        switch (kDistanceSampler)
        {
        case DistanceSampler::DeltaTracking:
            return sampleDistanceDeltaTracking(gridVolume, ray.origin, ray.dir, nearFar, tsg, ds);
        case DistanceSampler::DeltaTrackingLocalMajorant:
            return sampleDistanceDeltaTrackingLocalDDA(gridVolume, ray.origin, ray.dir, nearFar, tsg, ds);
        default:
            return false;
        }
    }

    float lookupDensityIndex(const GridVolume gridVolume, const float3 ipos, const float3 u, const Grid densityGrid, inout DensityAccessor densityAccessor)
    {
        logVolumeLookup();
#if GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
        return gridVolume.data.densityScale * densityGrid.lookupStochasticIndexTex(ipos, u);
#else
        return gridVolume.data.densityScale * densityGrid.lookupStochasticIndex(ipos, u, densityAccessor);
#endif
    }

    bool intersectVolume(const GridVolume gridVolume, const Ray ray, out float2 nearFar)
    {
        // Intersect with volume bounds and get intersection interval along the view ray.
        AABB bounds = gridVolume.getBounds();
        intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, nearFar);
        nearFar.x = max(nearFar.x, ray.tMin);
        nearFar.y = min(nearFar.y, ray.tMax);
        return nearFar.x < nearFar.y;
    }

    float evalTransmittanceDeltaTracking<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        // Transform to index-space.
        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);

        const float invMajorant = 1.f / (gridVolume.data.densityScale * densityGrid.getMaxValue());

        // Delta tracking.
        float t = nearFar.x;
        while (true)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajorant;
            if (t >= nearFar.y)
            {
                return 1.f;
            }
            const float d = lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);

            // Russian roulette.
            if (sampleNext1D(sg) < d * invMajorant) return 0.f;
        }
        return 1.f;
    }

    float evalTransmittanceRatioTracking<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);

        float Tr = 1.f;
        const float invMajorant = 1.f / (gridVolume.data.densityScale * densityGrid.getMaxValue());

        // Ratio tracking.
        float t = nearFar.x;
        while (true)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajorant;
            if (t >= nearFar.y)
            {
                return Tr;
            }
            const float d = lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);

            // The 'max' guards against non-bounding majorants which can happen due to density interpolation in NanoVDB.
            // They are rare but can lead to fireflies, and the max prevents them at the cost of (imperceptible) bias.
            Tr *= max(0.f, 1.f - d * invMajorant);
            if (Tr < 0.1f)
            {
                if (sampleNext1D(sg) >= Tr) return 0.f;
                Tr = 1.f;
            }
        }
        return Tr;
    }

    bool stepToNextCollision(const GridVolume gridVolume, const Grid densityGrid, const float3 ipos, const float3 idir, const float3 ri,
                             float tauToGo, inout float t, const float tFar, bool adaptive, out float majorant, inout DensityAccessor accessor)
    {
        // Warning: Changing the mip level based on 'tauToGo' below could potentially introduce bias, although it would not seem to be visible in practice.
        while (t < tFar)
        {
            // Core DDA step.
            const int mip = round(getAccessorLevel(accessor));
            const float3 pos = ipos + idir * t;
            const float tExit = stepDDA(ipos, ri, pos, mip);
            logVolumeLookup();
            majorant = gridVolume.data.densityScale * densityGrid.lookupIndexLocalMajorantTex(pos, mip);
            const float dt = tExit - t;
            t = tExit;
            // Test for (tentative) collisions.
            const float dtau = dt * majorant;
            tauToGo -= dtau;
            if (adaptive) increaseAccessorLevel(accessor);
            if (tauToGo > 0.f) continue;
            // Revert to collision.
            t += dt * tauToGo / dtau;
            if (adaptive) decreaseAccessorLevel(accessor);
            return (t < tFar);
        }
        return false;
    }

    float evalTransmittanceRatioTrackingLocalDDA<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);
        const float3 ri = 1.f / idir;
        // Ratio tracking.
        float Tr = 1.f;
        float t = nearFar.x;
        while (1)
        {
            float tauToGo = -log(1.f - sampleNext1D(sg)), majorant;
            if (!stepToNextCollision(gridVolume, densityGrid, ipos, idir, ri, tauToGo, t, nearFar.y, true, majorant, densityAccessor)) return Tr;
            const float d = lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);
            Tr *= max(0.f, 1.f - d / majorant);
            // Russian roulette.
            if (Tr < 0.1f)
            {
                if (sampleNext1D(sg) >= Tr) return 0.f;
                Tr = 1.f;
            }
        }
        return Tr;
    }

#ifdef FALCOR_INTERNAL
    float evalTransmittanceUnbiasedRayMarching<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        float samples[kUnbiasedRayMarchingMaxSamples];

        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        // Transform to index-space.
        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);

        // Compute the tuple size for the optical thickness samples.
        const float rayLength = nearFar.y - nearFar.x;
        const float majorantDensity = gridVolume.data.densityScale * densityGrid.getMaxValue();
        const float majorantIntegral = majorantDensity * rayLength;
        const uint tupleSize = UnbiasedRayMarchingRoulette::evalTupleSize(majorantIntegral * kUnbiasedRayMarchingLookupMultiplier); // Note: majorantIntegral - minorantIntegral if using local statistics.

        // Determine the Taylor expansion order.
        var roulette = UnbiasedRayMarchingRoulette(sampleNext1D(sg));
        uint maxOrder = 0;
        while (maxOrder < kUnbiasedRayMarchingMaxOrder)
        {
            if (roulette.accept(maxOrder + 1) == 0.f) break;
            maxOrder++;
        }

        // Evaluate the comb-filtered optical thickness samples. TODO: This is the piece of code that may slow the method down on GPU since nSamples may vary wildly. How to fix?
        float rouletteProbability = 1.f;
        float rouletteRandom;
        if (kRayMarchingRouletteThreshold > 0.f) rouletteRandom = sampleNext1D(sg);

        uint nSamples = maxOrder + 1;
        const float dt = rayLength / tupleSize;
        for (uint i = 0; i < nSamples; i++) {
            float integral = 0.f;
            float t = nearFar.x + sampleNext1D(sg) * dt;
            for (uint j = 0; j < tupleSize; j++) {
                integral += dt * lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);

                t += dt;

                // Experimental: Apply roulette to stop the integration early in dense media, but let the contributions still go to zero by making targetProbability only the square root.
                //               controlTransmittance = exp(-dt * integral);
                //               targetProbability = sqrt(controlTransmittance / kRayMarchingRouletteThreshold);
                if (kRayMarchingRouletteThreshold > 0.f && i == 0 && integral > -log(kRayMarchingRouletteThreshold)) // <=> rouletteProbability < 1
                {
                    rouletteProbability = exp(-0.5f * integral) / sqrt(kRayMarchingRouletteThreshold);
                    if (rouletteProbability < rouletteRandom) return 0.f;
                }
            }
            samples[i] = -integral;
        }

        // Evaluate the power series by using each sample as the pivot in turn, and evaluate the mean.
        float Tr = 0.f;
        for (uint pivotIndex = 0; pivotIndex < nSamples; pivotIndex++) {
            const float pivot = samples[pivotIndex];

            float m[kUnbiasedRayMarchingMaxOrder + 1];
            evalShiftedElementaryMeans(samples, maxOrder, pivotIndex, m);

            float taylorExpansion = 1.f;
            float c = 1.f;
            roulette.rewind();
            for (uint k = 1; k <= maxOrder; k++)
            {
                c /= k * roulette.accept(k);
                taylorExpansion += c * m[k];
            }
            Tr += exp(pivot) * taylorExpansion;
        }

        return Tr / nSamples / min(1.f, rouletteProbability);
    }

    float evalTransmittanceBiasedRayMarching<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        // Transform to index-space.
        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);

        // Automatic lookup count according to An unbiased ray-marching transmittance estimator (Kettunen et al. 2021).
        const float rayLength = nearFar.y - nearFar.x;
        const float majorantDensity = gridVolume.data.densityScale * densityGrid.getMaxValue();
        const float majorantIntegral = majorantDensity * rayLength;
        const uint lookupCount = (uint)evalRayMarchingTargetLookupCount(majorantIntegral * kBiasedRayMarchingLookupMultiplier);

        const float dt = rayLength / lookupCount;

        // Experimental: Apply roulette to stop the integration early in dense media, but let the contributions still go to zero by making survivalProbability only the square root.
        float rouletteCutOff;
        if (kRayMarchingRouletteThreshold > 0.f)
        {
            // survivalProbability = exp(-0.5f * dt * integral) / sqrt(kRayMarchingRouletteThreshold);
            //
            // survivalProbability < rouletteRandom
            //     <=> integral > -2 log(rouletteRandom * sqrt(kRayMarchingRouletteThreshold)) / dt
            //     <=> integral > rouletteCutOff
            float rouletteRandom = sampleNext1D(sg);
            rouletteCutOff = -2.f * log(rouletteRandom * sqrt(kRayMarchingRouletteThreshold)) / dt;
        }

        // Integrate.
        float integral = 0.f;
        float u = sampleNext1D(sg);
        float t = nearFar.x + u * dt;
        for (uint i = 0; i < lookupCount; i++) {
            integral += lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);
            t += dt;
            if (kRayMarchingRouletteThreshold > 0.f && integral > rouletteCutOff) return 0.f; // <=> survivalProbability < rouletteRandom
        }

        float survivalProbability = 1.f;
        if (kRayMarchingRouletteThreshold > 0.f) survivalProbability = min(1.f, exp(-0.5f * dt * integral) / sqrt(kRayMarchingRouletteThreshold));

        return exp(-dt * integral) / survivalProbability;
    }

    float evalTransmittanceBiasedRayMarchingLocalDDA<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        // Transform to index-space.
        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);
        const float3 ri = 1.f / idir;

        // Experimental: Apply roulette to stop the integration early in dense media, but let the contributions still go to zero by making survivalProbability only the square root.
        float rouletteCutOff;
        if (kRayMarchingRouletteThreshold > 0.f)
        {
            // survivalProbability = exp(-0.5f * kBiasedRayMarchingLocalCDFStepSize * integral) / sqrt(kRayMarchingRouletteThreshold);
            //
            // survivalProbability < rouletteRandom
            //     <=> integral > -2 log(rouletteRandom * sqrt(kRayMarchingRouletteThreshold)) / kBiasedRayMarchingLocalCDFStepSize
            //     <=> integral > rouletteCutOff
            float rouletteRandom = sampleNext1D(sg);
            rouletteCutOff = -2.f * log(rouletteRandom * sqrt(kRayMarchingRouletteThreshold)) / kBiasedRayMarchingLocalCDFStepSize;
        }

        // Integrate.
        float integral = 0.f;
        float u = sampleNext1D(sg);
        float t = nearFar.x;
        float tauToGo = kBiasedRayMarchingLocalCDFStepSize * u;

        while (true)
        {
            float majorant;
            if (!stepToNextCollision(gridVolume, densityGrid, ipos, idir, ri, tauToGo, t, nearFar.y, false, majorant, densityAccessor)) break;
            tauToGo = kBiasedRayMarchingLocalCDFStepSize;

            const float d = lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);
            integral += d / majorant;

            if (kRayMarchingRouletteThreshold > 0.f && integral > rouletteCutOff) return 0.f; // <=> survivalProbability < rouletteRandom
        }

        float survivalProbability = 1.f;
        if (kRayMarchingRouletteThreshold > 0.f) survivalProbability = min(1.f, exp(-0.5f * kBiasedRayMarchingLocalCDFStepSize * integral) / sqrt(kRayMarchingRouletteThreshold));

        return exp(-kBiasedRayMarchingLocalCDFStepSize * integral) / survivalProbability;
    }
#endif

    bool sampleDistanceDeltaTracking<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg, out DistanceSample ds)
    {
        ds = {};

        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);
        const float invMajorant = 1.f / (gridVolume.data.densityScale * densityGrid.getMaxValue());

        // Delta tracking.
        float t = nearFar.x;
        while (t < nearFar.y)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajorant;
            const float d = lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);

            // Scatter on real collision.
            if (sampleNext1D(sg) < d * invMajorant)
            {
                ds.t = t;
                ds.thp = gridVolume.data.albedo;
                return true;
            }
        }

        return false;
    }

    bool sampleDistanceDeltaTrackingLocalDDA<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg, out DistanceSample ds)
    {
        ds = {};
        ds.thp = 1.f;

        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);

        // Init DDA.
        const float3 ri = 1.f / idir;
        float t = nearFar.x;
        while (1)
        {
            float tauToGo = -log(1.f - sampleNext1D(sg)), majorant;
            if (!stepToNextCollision(gridVolume, densityGrid, ipos, idir, ri, tauToGo, t, nearFar.y, true, majorant, densityAccessor)) return false;
            // Classify collision as real or null.
            if (sampleNext1D(sg) * majorant < lookupDensityIndex(gridVolume, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor))
            {
                // Real collision, terminate.
                ds.t = t;
                ds.thp = gridVolume.data.albedo;
                return true;
            }
            // Null collision, continue with new sample.
        }
        return false;
    }
};
