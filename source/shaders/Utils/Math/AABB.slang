/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

/**
 * Axis-aligned bounding box (AABB).
 */
struct AABB
{
    float3 minPoint; ///< Minimum point.
    float3 maxPoint; ///< Maximum point. If any minPoint > maxPoint the box is invalid.

    /**
     * Initializes an AABB.
     * Note if minPoint > maxPoint in any component the box is invalid.
     */
    __init(float3 minPoint, float3 maxPoint)
    {
        this.minPoint = minPoint;
        this.maxPoint = maxPoint;
    }

    /**
     * Set box to single point.
     */
    [mutating]
    void set(float3 p)
    {
        minPoint = p;
        maxPoint = p;
    }

    /**
     * Set the box corners explicitly.
     * Note if min > max in any component the box is invalid.
     */
    [mutating]
    void set(float3 _min, float3 _max)
    {
        minPoint = _min;
        maxPoint = _max;
    }

    /**
     * Invalidates the box.
     */
    [mutating]
    void invalidate()
    {
        minPoint = FLT_MAX;
        maxPoint = -FLT_MAX;
    }

    /**
     * Returns true if the box is valid.
     */
    bool valid() { return all(minPoint <= maxPoint); }

    /**
     * Grows the box to include the point p.
     */
    [mutating]
    void include(float3 p)
    {
        minPoint = min(minPoint, p);
        maxPoint = max(maxPoint, p);
    }

    /**
     * Grows the box to include another box.
     */
    [mutating]
    void include(AABB b)
    {
        minPoint = min(minPoint, b.minPoint);
        maxPoint = max(maxPoint, b.maxPoint);
    }

    /**
     * Check if point is included in the box.
     * @return True if p is in the box (inclusive test), false if outside or box invalid.
     */
    bool contains(float3 p) { return valid() && all(p >= minPoint && p <= maxPoint); }

    /**
     * Returns the box center.
     * @return Center of the box if valid, undefined otherwise.
     */
    float3 center() { return (minPoint + maxPoint) * 0.5f; }

    /**
     * Returns the box extent.
     * @return Size of the box if valid, undefined otherwise.
     */
    float3 extent() { return maxPoint - minPoint; }

    /**
     * Returns the surface area of the box.
     * @return Surface area if box is valid, undefined otherwise.
     */
    float area()
    {
        float3 e = extent();
        return (e.x * e.y + e.x * e.z + e.y * e.z) * 2.f;
    }

    /**
     * Return the volume of the box.
     * @return Volume if the box is valid, undefined otherwise.
     */
    float volume()
    {
        float3 e = extent();
        return e.x * e.y * e.z;
    }

    /**
     * Returns the radius of the minimal sphere that encloses the box.
     * @return Radius of minimal bounding sphere, or undefined if box is invalid.
     */
    float radius() { return 0.5f * length(extent()); }

    /**
     * Check if two boxes intersect. The test is inclusive on both sides.
     * @param[in] other The other box.
     * @return True if the two boxes intersect. The result is undefined if either box is invalid.
     */
    bool intersects(const AABB other) { return all(maxPoint >= other.minPoint && minPoint <= other.maxPoint); }

    /**
     * Returns the minimum distance between the box and a point.
     * The distance is the minimum distance between any point in the box and the other point.
     * @param[in] p The point to which to compute the distance.
     * @return Minimum distance between between point and box, or 0 if the point lies inside. The result is undefined if the box is invalid.
     */
    float minDistance(float3 p)
    {
        float3 d1 = minPoint - p;
        float3 d2 = p - maxPoint;
        float3 d = max(max(d1, d2), 0.f);
        return length(d);
    }

    /**
     * Returns the minimum distance between two boxes.
     * The distance is the minimum distance between any two points in the boxes.
     * @param[in] other The other box.
     * @return Minimum distance between boxes, or 0 if they intersect. The result is undefined if either box is invalid.
     */
    float minDistance(const AABB other)
    {
        float3 d1 = minPoint - other.maxPoint;
        float3 d2 = other.minPoint - maxPoint;
        float3 d = max(max(d1, d2), 0.f);
        return length(d);
    }
};
