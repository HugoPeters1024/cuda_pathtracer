#ifndef H_TYPES
#define H_TYPES

#include <vector>
#include <algorithm>
#include <stack>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_object_loader.h"
#include "use_cuda.h"
#include <limits>
#include <vector_functions.h>

#define inf 99999999

#ifdef __CUDACC__
#define HYBRID __host__ __device__
#else
#define HYBRID
#endif 


// Asserts that the current location is within the space
#define CUDA_LIMIT(x,y) { if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return; }

struct Sphere
{
    float3 pos;
    float radius;
    float3 color;

    HYBRID inline float3 centroid() { return pos; }
};

struct Box
{
    float3 vmin;
    float3 vmax;

    HYBRID inline float3 centroid() const { return (vmin + vmax) * 0.5; }
    HYBRID inline float diagonal() const { return length(vmin - vmax); };
    inline float volume() const { return abs((vmax.x - vmin.x) * (vmax.y - vmin.y) * (vmax.z - vmin.z)); }

    bool overlaps(const Box& other) const {
        return vmax.x >= other.vmin.x && other.vmax.x >= vmin.x &&
               vmax.y >= other.vmin.y && other.vmax.y >= vmin.y &&
               vmax.z >= other.vmin.z && other.vmax.z >= vmin.z;
    }
};

struct Ray
{
    float3 origin;
    float3 direction;
    float3 invdir;
    int signs[3];
};

struct HitInfo
{
    bool intersected;
    uint triangle_id;
    float t;
    float3 normal;
    float3 pos;
};

struct Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;
    float3 n0;
    float3 n1;
    float3 n2;
    float3 color;

    HYBRID inline float3 centroid() const { return (v0 + v1 + v2) / 3.0f; }
    HYBRID inline float max_x() const { return max(v0.x, max(v1.x, v2.x)); }
    HYBRID inline float max_y() const { return max(v0.y, max(v1.y, v2.y)); }
    HYBRID inline float max_z() const { return max(v0.z, max(v1.z, v2.z)); }
    HYBRID inline float min_x() const { return min(v0.x, min(v1.x, v2.x)); }
    HYBRID inline float min_y() const { return min(v0.y, min(v1.y, v2.y)); }
    HYBRID inline float min_z() const { return min(v0.z, min(v1.z, v2.z)); }

    Box getBoundingBox() const
    {
        float vminx = min(v0.x, min(v1.x, v2.x));
        float vminy = min(v0.y, min(v1.y, v2.y));
        float vminz = min(v0.z, min(v1.z, v2.z));

        float vmaxx = max(v0.x, max(v1.x, v2.x));
        float vmaxy = max(v0.y, max(v1.y, v2.y));
        float vmaxz = max(v0.z, max(v1.z, v2.z));

        return Box {
            make_float3(vminx, vminy, vminz),
            make_float3(vmaxx, vmaxy, vmaxz)
        };
    }

};

static bool __compare_triangles_x (Triangle a, Triangle b) {
    return (a.centroid().x < b.centroid().x);
}

static bool __compare_triangles_y (Triangle a, Triangle b) {
    return (a.centroid().y < b.centroid().y);
}

static bool __compare_triangles_z (Triangle a, Triangle b) {
    return (a.centroid().z < b.centroid().z);
}

struct BVHTree
{
    BVHTree* child1;
    BVHTree* child2;
    bool isLeaf;
    std::vector<Triangle> triangles;
    Box boundingBox;
    int used_level = -1;

    uint treeSize() const {
        if (isLeaf) return 1;
        uint left = child1 != nullptr ? child1->treeSize() : 0;
        uint right = child2 != nullptr ? child2->treeSize() : 0;
        return 1 + left + right;
    }
};

struct BVHNode
{
    Box boundingBox;
    uint parent;
    uint child1;
    uint child2;
    uint split_plane;
    uint t_start;
    uint t_count;
};



#endif
