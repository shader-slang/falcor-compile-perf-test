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
#include "Scene/SDFs/SDFGridNoDefines.slangh"

 /** The scene sets the SCENE_SDF_GRID_IMPLEMENTATION define to select what SDFGridHitData implementation should be used.
 */

struct SDFGridHitData
{
#ifdef FALCOR_INTERNAL
#if SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_NDSDF
    uint2 locationCode;                 ///< Contains excluded voxel unit component, grid level, and voxel coordinates.
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVS
    uint primitiveID;                   ///< Contains excluded voxel unit component and voxel primitive ID.
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SBS
#if SCENE_SDF_SBS_VIRTUAL_BRICK_COORDS_BITS + SCENE_SDF_SBS_BRICK_LOCAL_VOXEL_COORDS_BITS <= 30
    uint packedVoxelCoords;             ///< Contains excluded voxel unit component, brick local voxel coordinates, and virtual brick coords.
#else
    uint packedVirtualBrickCoords;      ///< Virtual brick coords.
    uint packedBrickLocalVoxelCoords;   ///< Contains excluded voxel unit component and brick local voxel coordinates.
#endif
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVO
    uint svoIndex;                      ///< Contains SVO index (similar to primitive ID).
#endif

    uint packedVoxelUnitCoords;         ///< Contains the two included voxel unit coordinate components as 16-bit unorms.
#else // FALCOR_INTERNAL
#if SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_NDSDF
    uint lod;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVS || SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SBS
    uint primitiveID;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVO
    uint svoIndex;
#endif

    float hitT;
#endif // FALCOR_INTERNAL
};
