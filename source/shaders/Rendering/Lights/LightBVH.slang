/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;
__exported import LightBVHTypes;

/** Light BVH data structure.
*/
struct LightBVH
{
    [root] StructuredBuffer<PackedNode> nodes;      ///< Buffer containing all the nodes from the BVH, with the root node located at index 0.
    StructuredBuffer<uint> triangleIndices;         ///< Buffer containing the indices of all emissive triangles. Each leaf node refers to a contiguous range of indices.
    StructuredBuffer<uint2> triangleBitmasks;       ///< Buffer containing for each emissive triangle, a bit mask of the traversal to follow in order to reach that triangle. Size: lights.triangleCount * sizeof(uint64_t).

    bool isLeaf(uint nodeIndex)
    {
        return nodes[nodeIndex].isLeaf();
    }

    LeafNode getLeafNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getLeafNode();
    }

    InternalNode getInternalNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getInternalNode();
    }

    SharedNodeAttributes getNodeAttributes(uint nodeIndex)
    {
        return nodes[nodeIndex].getNodeAttributes();
    }

    uint getNodeTriangleIndex(const LeafNode node, uint index)
    {
        return triangleIndices[node.triangleOffset + index];
    }
};

/** Variant of the Light BVH data structure where the nodes are writable.
    This is used for BVH refit.
    TODO: Refactor to avoid code duplication using future Slang features.
*/
struct RWLightBVH
{
    [root] RWStructuredBuffer<PackedNode> nodes;    ///< Buffer containing all the nodes from the BVH, with the root node located at index 0.
    StructuredBuffer<uint> triangleIndices;         ///< Buffer containing the indices of all emissive triangles. Each leaf node refers to a contiguous range of indices.
    StructuredBuffer<uint2> triangleBitmasks;       ///< Buffer containing for each emissive triangle, a bit mask of the traversal to follow in order to reach that triangle. Size: lights.triangleCount * sizeof(uint64_t).

    bool isLeaf(uint nodeIndex)
    {
        return nodes[nodeIndex].isLeaf();
    }

    LeafNode getLeafNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getLeafNode();
    }

    InternalNode getInternalNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getInternalNode();
    }

    uint getNodeTriangleIndex(const LeafNode node, uint index)
    {
        return triangleIndices[node.triangleOffset + index];
    }

    // Setters

    void setLeafNode(uint nodeIndex, LeafNode node)
    {
        nodes[nodeIndex].setLeafNode(node);
    }

    void setInternalNode(uint nodeIndex, InternalNode node)
    {
        nodes[nodeIndex].setInternalNode(node);
    }
};
