/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Math.PackedFormats;
__exported import Scene.HitInfo;
__exported import Utils.Math.Ray;
__exported import Utils.Sampling.SampleGenerator;
__exported import Rendering.Materials.InteriorList;
__exported import RenderPasses.Shared.Denoising.NRDData;
__exported import GuideData;
__exported import NRCData;
__exported import QueueData;

static const uint kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to avoid getting stuck in pathological cases.

static const float kRayTMax = 1e30f;

// Be careful with changing these. PathFlags share 32-bit uint with vertexIndex. For now, we keep 10 bits for vertexIndex.
// PathFlags take higher bits, VertexIndex takes lower bits.
static const uint kVertexIndexBitCount = 10u;
static const uint kVertexIndexBitMask = (1u << kVertexIndexBitCount) - 1u;
static const uint kPathFlagsBitCount = 32u - kVertexIndexBitCount;
static const uint kPathFlagsBitMask = ((1u << kPathFlagsBitCount) - 1u) << kVertexIndexBitCount;

/** Path flags. The path flags are currently stored in kPathFlagsBitCount bits.
*/
enum class PathFlags
{
    active                      = 0x0001,   ///< Path is active/terminated.
    hit                         = 0x0002,   ///< Result of the scatter ray (0 = miss, 1 = hit).

    transmission                = 0x0004,   ///< Scatter ray went through a transmission event.
    specular                    = 0x0008,   ///< Scatter ray went through a specular event.
    delta                       = 0x0010,   ///< Scatter ray went through a delta event.
    volume                      = 0x0020,   ///< Scatter ray went through a volume event.

    insideDielectricVolume      = 0x0040,   ///< Path vertex is inside a dielectric volume.
    lightSampledUpper           = 0x0080,   ///< Last path vertex sampled lights using NEE (in upper hemisphere).
    lightSampledLower           = 0x0100,   ///< Last path vertex sampled lights using NEE (in lower hemisphere).

    diffusePrimaryHit           = 0x0200,   ///< Primary hit was sampled from the diffuse BSDF lobe.
    specularPrimaryHit          = 0x0400,   ///< Primary hit was sampled from the specular BSDF lobe.
    deltaReflectionPrimaryHit   = 0x0800,   ///< Primary hit was sampled as the delta reflection.
    deltaTransmissionPath       = 0x1000,   ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
    deltaOnlyPath               = 0x2000,   ///< There was no non-delta events along the path so far.

    hasVolumeProperties         = 0x4000    ///< Indicates that we take volume parameters from the path. Note: This is olnly sigmaA/S for now, not the phase function.

    // Bits 15 to kPathFlagsBitCount are still unused.
};

/** Bounce types. We keep separate counters for all of these.
*/
enum class BounceType
{
    Diffuse                 = 0,    ///< Diffuse reflection.
    Specular                = 1,    ///< Specular reflection (including delta).
    Transmission            = 2,    ///< Transmission (all kinds).
    Volume                  = 3,    ///< Scatter event in heterogeneous volume.
};

/** Packed path state. Contains all data from PathState in packed form.
    Default total size: 80 + (8-16) + (2*4) + (4-16) bytes.
*/
struct PackedPathState
{
    uint4 packed[5];
    PackedHitInfo hit;
    GuideData guideData;        ///< Denoiser guide data.
    InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties. The size depends on the config (4B/item).
    SampleGenerator sg;         ///< Sample generator state. Typically 4-16B.
};

struct ReSTIRGIData
{
    float3 creationPoint;       ///< Position of visible point.
    float3 creationNormal;      ///< Normal of visible point.
    float3 position;            ///< Position of secondary ray's hit point.
    float3 normal;              ///< Normal of secondary ray's hit point.
    float3 L0;                  ///< Direct lighting.
    bool   isDisabled;          ///< A flag showing if a pixel is calculated using ReSTIR GI.
};

// TODO: Compact encoding to reduce live registers, e.g. packed HitInfo, packed normals.
/** Live state for the path tracer.
*/
struct PathState
{
    uint        id;                     ///< Path ID encodes (pixel, sampleIdx) with 12 bits each for pixel x|y and 8 bits for sample index.

    uint        flagsAndVertexIndex;    ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
                                        ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    uint16_t    rejectedHits;           ///< Number of false intersections rejected along the path. This is used as a safeguard to avoid deadlock in pathological cases.
    float16_t   sceneLength;            ///< Path length in scene units (0.f at primary hit).
    uint        bounceCounters;         ///< Packed counters for different types of bounces (see BounceType).

    // Scatter ray
    float3      origin;                 ///< Origin of the scatter ray.
    float3      dir;                    ///< Scatter ray normalized direction.
    float       pdf;                    ///< Pdf for generating the scatter ray.
    float3      normal;                 ///< Shading normal at the scatter ray origin.
    HitInfo     hit;                    ///< Hit information for the scatter ray. This is populated at committed triangle hits.

    float3      sigmaA;                 ///< SigmaA taken from e.g the material instance
    float3      sigmaS;                 ///< SigmaS taken from e.g. the material instance

    float3      thp;                    ///< Path throughput.
    float3      L;                      ///< Accumulated path contribution.

    GuideData   guideData;              ///< Denoiser guide data.
    NRCData     nrcData;                ///< Neural radiance cache data.
    float4x4    imageXform;             ///< Accumulated image transform along the path. It's used only for NRD guide paths.
    InteriorList interiorList;          ///< Interior list. Keeping track of a stack of materials with medium properties.
    SampleGenerator sg;                 ///< Sample generator state. Typically 4-16B.
    ReSTIRGIData reSTIRGIData;          ///< ReSTIR GI data.

    // Accessors
    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags::active); }
    bool isHit() { return hasFlag(PathFlags::hit); }
    bool isTransmission() { return hasFlag(PathFlags::transmission); }
    bool isSpecular() { return hasFlag(PathFlags::specular); }
    bool isDelta() { return hasFlag(PathFlags::delta); }
    bool isVolume() { return hasFlag(PathFlags::volume); }
    bool isInsideDielectricVolume() { return hasFlag(PathFlags::insideDielectricVolume); }

    bool isLightSampled()
    {
        const uint bits = (uint(PathFlags::lightSampledUpper) | uint(PathFlags::lightSampledLower)) << kVertexIndexBitCount;
        return flagsAndVertexIndex & bits;
    }

    bool isLightSampledUpper() { return hasFlag(PathFlags::lightSampledUpper); }
    bool isLightSampledLower() { return hasFlag(PathFlags::lightSampledLower); }
    bool isDiffusePrimaryHit() { return hasFlag(PathFlags::diffusePrimaryHit); }
    bool isSpecularPrimaryHit() { return hasFlag(PathFlags::specularPrimaryHit); }
    bool isDeltaReflectionPrimaryHit() { return hasFlag(PathFlags::deltaReflectionPrimaryHit); }
    bool isDeltaTransmissionPath() { return hasFlag(PathFlags::deltaTransmissionPath); }
    bool isDeltaOnlyPath() { return hasFlag(PathFlags::deltaOnlyPath); }

    bool hasVolumeProperties() { return hasFlag(PathFlags::hasVolumeProperties); }

    // Check if the scatter event is samplable by the light sampling technique.
    bool isLightSamplable() { return !isDelta(); }

    [mutating] void terminate() { setFlag(PathFlags::active, false); }
    [mutating] void setActive() { setFlag(PathFlags::active); }
    [mutating] void setHit(HitInfo hitInfo) { hit = hitInfo; setFlag(PathFlags::hit); }
    [mutating] void clearHit() { setFlag(PathFlags::hit, false); }

    [mutating] void clearEventFlags()
    {
        const uint bits = (uint(PathFlags::transmission) | uint(PathFlags::specular) | uint(PathFlags::delta) | uint(PathFlags::volume)) << kVertexIndexBitCount;
        flagsAndVertexIndex &= ~bits;
    }

    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    [mutating] void setVolume(bool value = true) { setFlag(PathFlags::volume, value); }
    [mutating] void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags::insideDielectricVolume, value); }
    [mutating] void setLightSampled(bool upper, bool lower) { setFlag(PathFlags::lightSampledUpper, upper); setFlag(PathFlags::lightSampledLower, lower); }
    [mutating] void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags::diffusePrimaryHit, value); }
    [mutating] void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags::specularPrimaryHit, value); }
    [mutating] void setDeltaReflectionPrimaryHit(bool value = true) { setFlag(PathFlags::deltaReflectionPrimaryHit, value); }
    [mutating] void setDeltaTransmissionPath(bool value = true) { setFlag(PathFlags::deltaTransmissionPath, value); }
    [mutating] void setDeltaOnlyPath(bool value = true) { setFlag(PathFlags::deltaOnlyPath, value); }
    [mutating] void setHasVolumeProperties(bool value = true) { setFlag(PathFlags::hasVolumeProperties, value); }

    bool hasFlag(PathFlags flag)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        return (flagsAndVertexIndex & bit) != 0;
    }

    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        if (value) flagsAndVertexIndex |= bit;
        else flagsAndVertexIndex &= ~bit;
    }

    uint getBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        return (bounceCounters >> shift) & 0xff;
    }

    [mutating] void setBounces(BounceType type, uint bounces)
    {
        const uint shift = (uint)type << 3;
        bounceCounters = (bounceCounters & ~((uint)0xff << shift)) | ((bounces & 0xff) << shift);
    }

    [mutating] void incrementBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        // We assume that bounce counters cannot overflow.
        bounceCounters += (1 << shift);
    }

    uint2 getPixel() { return uint2(id, id >> 12) & 0xfff; }
    uint getSampleIdx() { return id >> 24; }

    // Unsafe - assumes that index is small enough.
    [mutating] void setVertexIndex(uint index)
    {
        // Clear old vertex index.
        flagsAndVertexIndex &= kPathFlagsBitMask;
        // Set new vertex index (unsafe).
        flagsAndVertexIndex |= index;
    }

    uint getVertexIndex() { return flagsAndVertexIndex & kVertexIndexBitMask; }

    // Unsafe - assumes that vertex index never overflows.
    [mutating] void incrementVertexIndex() { flagsAndVertexIndex += 1; }
    // Unsafe - assumes that vertex index will never be decremented below zero.
    [mutating] void decrementVertexIndex() { flagsAndVertexIndex -= 1; }

    Ray getScatterRay()
    {
        return Ray(origin, dir, 0.f, kRayTMax);
    }

    // Helpers for packing the full path state.

    PackedPathState encode()
    {
        PackedPathState s = {};

        s.packed[0].xyz = asuint(origin);
        s.packed[0].w = id;

        s.packed[1].xyz = asuint(dir);
        s.packed[1].w = flagsAndVertexIndex;

        s.packed[2].xyz = asuint(normal);
        s.packed[2].w = uint(rejectedHits) | ((f32tof16(sceneLength) & 0xffff) << 16);

        s.packed[3].xyz = asuint(thp);
        s.packed[3].w = bounceCounters;

        s.packed[4].xyz = asuint(L);
        s.packed[4].w = asuint(pdf);

        s.hit = hit.getData();
        s.guideData = guideData;
        s.interiorList = interiorList;
        s.sg = sg;

        return s;
    }

    [mutating] void decode(const PackedPathState s)
    {
        this = {};

        origin = asfloat(s.packed[0].xyz);
        id = s.packed[0].w;

        dir = asfloat(s.packed[1].xyz);
        flagsAndVertexIndex = s.packed[1].w;

        normal = asfloat(s.packed[2].xyz);
        rejectedHits = uint16_t(s.packed[2].w & 0xffff);
        sceneLength = float16_t(f16tof32(s.packed[2].w >> 16));

        thp = asfloat(s.packed[3].xyz);
        bounceCounters = s.packed[3].w;

        L = asfloat(s.packed[4].xyz);
        pdf = asfloat(s.packed[4].w);

        hit = HitInfo(s.hit);
        guideData = s.guideData;
        interiorList = s.interiorList;
        sg = s.sg;
    }

    // Helpers for packing the path state for a specular path.
    // These are used when pushing a specular path onto another queue.
    // Note that not all fields are stored/restored, but only the necessary ones.

    PackedSpecularPath encodeSpecularPath()
    {
        PackedSpecularPath s = {};

        s.packed[0].xyz = asuint(origin);
        s.packed[0].w = encodeNormal2x16(dir);

        s.packed[1].xyz = asuint(thp); // TODO: 4B/8B LogYuv
        s.packed[1].w = id;

        s.packed[2].xyz = asuint(L); // TODO: 4B/8B LogYuv or write directly to output buffer
        s.packed[2].w = flagsAndVertexIndex;

        s.packedHit = hit.getData();
        s.interiorList = interiorList; // TODO: Remove if we handle dielectrics fully.
        s.sg = sg;
        s.bounceCounters = (getBounces(BounceType::Specular)) | (getBounces(BounceType::Transmission) << 8);
        s.sceneLength = sceneLength;

        return s;
    }

    [mutating] void decodeSpecularPath(const PackedSpecularPath s)
    {
        this = {};

        origin = asfloat(s.packed[0].xyz);
        dir = decodeNormal2x16(s.packed[0].w);

        thp = asfloat(s.packed[1].xyz);
        id = s.packed[1].w;

        L = asfloat(s.packed[2].xyz);
        flagsAndVertexIndex = s.packed[2].w;

        hit = HitInfo(s.packedHit);
        interiorList = s.interiorList;
        sg = s.sg;
        setBounces(BounceType::Specular, s.bounceCounters & 0xff);
        setBounces(BounceType::Transmission, (s.bounceCounters >> 8) & 0xff);
        sceneLength = s.sceneLength;
    }
};
