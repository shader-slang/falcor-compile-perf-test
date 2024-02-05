/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Set of path coherence hints used for reordering.
*/
struct PathCoherenceHints
{
    bool terminatedByRussianRoulette;   ///< handleHit() will be terminated early due to russian roulette.
    bool terminated;                    ///< handleHit() will be terminated early.
    bool samplesLight;                  ///< handleHit() will sample a light.
    bool computesEmissive;              ///< handleHit() will compute emission.
};
