/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Scene.HitInfo;
__exported import Utils.Math.Ray;

// TODO: Use default arguments when slang supports them:
// bool traceRay(const RayDesc ray, out HitInfo hit, out float hitT, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff);
// bool traceVisibilityRay(const RayDesc ray, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff);
// https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1179

/** Interface for scene ray queries.
*/
interface ISceneRayQuery
{
    /** Trace a ray against the scene and return the closest hit point.
        \param[in] ray Ray.
        \param[out] hit Hit info (only valid if function returns true).
        \param[out] hitT Hit distance (only valid if function returns true).
        \param[in] rayFlags Optional ray flags OR'ed with queries flags.
        \param[in] instanceInclusionMask Includes/rejects geometry based on instance mask.
        \return Returns true if the ray does intersect the scene.
    */
    bool traceRay(const Ray ray, out HitInfo hit, out float hitT, uint rayFlags, uint instanceInclusionMask);

    /** Trace a visibility ray against the scene.
        \param[in] ray Ray.
        \param[in] rayFlags Optional ray flags OR'ed with queries flags.
        \param[in] instanceInclusionMask Includes/rejects geometry based on instance mask.
        \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
    */
    bool traceVisibilityRay(const Ray ray, uint rayFlags, uint instanceInclusionMask);
};
