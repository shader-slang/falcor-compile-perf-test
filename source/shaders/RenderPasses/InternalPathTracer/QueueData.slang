/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Scene.HitInfo;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.InteriorList;

// TODO: 16B aligned
/** Packed path state for specular paths.
*/
struct PackedSpecularPath
{
    uint4 packed[3];

    PackedHitInfo packedHit;    ///< Packed hit information. Size 8-12B depending on scene.
    InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties. The size depends on the config (4B/item).
    SampleGenerator sg;         ///< Sample generator state. Typically 4-16B.
    uint16_t bounceCounters;    ///< Packed counters for different types of bounces (only BounceType::Specular and BounceType::Transmission).
    float16_t sceneLength;      ///< Path length in scene units (0.f at primary hit).
};
