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
import Utils.Math.FormatConversion;
import Utils.Sampling.SampleGenerator;

/** Per-sample color data.
 */
struct ColorData
{
    uint data;

    float3 get() { return decodeLogLuvHDR(data); }

    [mutating]
    void set(float3 color) { data = encodeLogLuvHDR(color); }
};

enum class PathFlags
{
    active = 0x0001,
    free = 0x0002,
};

/** Additional state needed to represent a path.
 */
struct PathState
{
    float3 wpos;
    uint id;
    float3 wdir;
    uint flags;
    float3 radiance;
    int length;
    float3 throughput;
    float pdf;
    SampleGenerator sg;

    static uint encodeID(const uint2 pixel, const uint sample) { return (sample << 24) | (pixel.y << 12) | pixel.x; }
    static uint2 decodePixel(const uint id) { return uint2(id, id >> 12) & 0xfff; }
    static uint decodeSample(const uint id) { return id >> 24; }

    uint2 getPixel() { return decodePixel(id); }
    uint getSample() { return decodeSample(id); }

    bool isTerminated() { return flags == 0; }
    bool isActive() { return flags != 0; }
    bool isFree() { return hasFlag(PathFlags::free); }

    [mutating]
    void terminate() { flags = 0; }
    [mutating]
    void setActive() { setFlag(PathFlags::active); }
    [mutating]
    void setFree() { setFlag(PathFlags::free); }

    bool hasFlag(PathFlags flag) { return (flags & uint(flag)) != 0; }
    [mutating]
    void setFlag(PathFlags flag, bool value = true)
    {
        if (value)
            flags |= uint(flag);
        else
            flags &= ~uint(flag);
    }
};
