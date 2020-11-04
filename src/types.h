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
    HYBRID inline float volume() const { return abs((vmax.x - vmin.x) * (vmax.y - vmin.y) * (vmax.z - vmin.z)); }
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
    const Box* collider;
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

    HYBRID inline float3 centroid() const { return (v0 + v1 + v2) * 0.33333333f; }
    HYBRID inline float max_x() const { return max(v0.x, max(v1.x, v2.x)); }
    HYBRID inline float max_y() const { return max(v0.y, max(v1.y, v2.y)); }
    HYBRID inline float max_z() const { return max(v0.z, max(v1.z, v2.z)); }
    HYBRID inline float min_x() const { return min(v0.x, min(v1.x, v2.x)); }
    HYBRID inline float min_y() const { return min(v0.y, min(v1.y, v2.y)); }
    HYBRID inline float min_z() const { return min(v0.z, min(v1.z, v2.z)); }

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

Box buildTriangleBox(const std::vector<Triangle> triangles)
{
    float min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = 100000;
    max_x = max_y = max_z = -100000;
    for(const Triangle t: triangles)
    {
        min_x = std::min(min_x, t.min_x());
        min_y = std::min(min_y, t.min_y());
        min_z = std::min(min_z, t.min_z());

        max_x = std::max(max_x, t.max_x());
        max_y = std::max(max_y, t.max_y());
        max_z = std::max(max_z, t.max_z());
    }

    return Box {
        make_float3(min_x, min_y, min_z), 
        make_float3(max_x, max_y, max_z)
    };
}

BVHTree* createBVH(std::vector<Triangle> triangles)
{
    BVHTree* ret = new BVHTree();
    ret->isLeaf = false;
    ret->child1 = nullptr;
    ret->child2 = nullptr;
    ret->boundingBox = buildTriangleBox(triangles);

    if (triangles.size() < 2)
    {
        ret->triangles = triangles;
        ret->isLeaf = true;
        return ret;
    }

    float min_cost = std::numeric_limits<float>::max();
    int min_level;

    // iterate over the 3 dimensions to find the most profitable splitting plane.
    for(int level=0; level<3; level++)
    {
        switch(level) {
            case 0: std::sort(triangles.begin(), triangles.end(), __compare_triangles_x); break;
            case 1: std::sort(triangles.begin(), triangles.end(), __compare_triangles_y); break;
            case 2: std::sort(triangles.begin(), triangles.end(), __compare_triangles_z); break;
        }

        int m = triangles.size() / 2;
        std::vector<Triangle> lows(triangles.begin(), triangles.begin() + m);
        std::vector<Triangle> highs(triangles.begin() + m, triangles.end());

        float cost1 = buildTriangleBox(lows).volume() * lows.size();
        float cost2 = buildTriangleBox(lows).volume() * lows.size();
        float cost = cost1 + cost2;

        if (cost < min_cost)
        {
            min_cost = cost;
            min_level = level;
        }
    }

    // Cost of not splittiing
    float currentCost = ret->boundingBox.volume() * triangles.size();
    // Split costs more
    if (min_cost >= currentCost)
    {
        ret->isLeaf = true;
        ret->triangles = triangles;
        return ret;
    }

    ret->used_level = min_level;

    // Sort the triangles one last time based on min_level
    switch (min_level) {
        case 0: std::sort(triangles.begin(), triangles.end(), __compare_triangles_x); break;
        case 1: std::sort(triangles.begin(), triangles.end(), __compare_triangles_y); break;
        case 2: std::sort(triangles.begin(), triangles.end(), __compare_triangles_z); break;
    }

    int m = triangles.size() / 2;
    std::vector<Triangle> lows(triangles.begin(), triangles.begin() + m);
    std::vector<Triangle> highs(triangles.begin() + m, triangles.end());
    if (lows.size() > 0) ret->child1 = createBVH(lows);
    if (highs.size() > 0) ret->child2 = createBVH(highs);
    return ret;
}

void sequentializeBvh(const BVHTree* root, std::vector<Triangle>& newTriangles, std::vector<BVHNode>& seqBvh)
{
    // Keep track of a parent id and subtree.
    std::stack<std::pair<uint, const BVHTree*>> work;
    work.push(std::pair<uint, const BVHTree*>(0, root));

    while(!work.empty())
    {
        std::pair<uint, const BVHTree*> tmp = work.top();
        work.pop();

        const BVHTree* currentNode = tmp.second;
        uint discoveredBy = tmp.first;

        BVHNode node;
        node.boundingBox = currentNode->boundingBox;
        node.parent = discoveredBy;
        node.split_plane = currentNode->used_level;

        // The node adds it's segment
        node.t_start = newTriangles.size();
        node.t_count = currentNode->triangles.size();

        // Add the triangles
        for (const Triangle& t : currentNode->triangles) newTriangles.push_back(t);


        // Calculate the indices of the children before hand
        uint myId = seqBvh.size();
        node.child1 = currentNode->child1 == nullptr ? 0 : myId + 1;
        node.child2 = currentNode->child2 == nullptr ? 0 : myId + 1 + currentNode->child1->treeSize();

        seqBvh.push_back(node);

        if (currentNode->child1 != nullptr) work.push(std::make_pair(myId, currentNode->child1));
        if (currentNode->child2 != nullptr) work.push(std::make_pair(myId, currentNode->child2));
    }
}

class Scene
{
public:
    std::vector<Triangle> triangles;
    void addModel(std::string filename, float3 color, float scale, float3 offset)
    {
        printf("Loading model %s\n", filename.c_str());
        tinyobj::ObjReaderConfig objConfig;
        objConfig.vertex_color = false;
        tinyobj::ObjReader objReader;
        objReader.ParseFromFile(filename, objConfig);

        for(int i=0; i<objReader.GetShapes()[0].mesh.indices.size(); i+=3)
        {
            auto it0 = objReader.GetShapes()[0].mesh.indices[i+0];
            auto it1 = objReader.GetShapes()[0].mesh.indices[i+1];
            auto it2 = objReader.GetShapes()[0].mesh.indices[i+2];
            auto vertices = objReader.GetAttrib().vertices;
            float3 v0 = offset + scale * make_float3(vertices[it0.vertex_index * 3 + 0], vertices[it0.vertex_index * 3 + 1], vertices[it0.vertex_index * 3 + 2]);
            float3 v1 = offset + scale * make_float3(vertices[it1.vertex_index * 3 + 0], vertices[it1.vertex_index * 3 + 1], vertices[it1.vertex_index * 3 + 2]);
            float3 v2 = offset + scale * make_float3(vertices[it2.vertex_index * 3 + 0], vertices[it2.vertex_index * 3 + 1], vertices[it2.vertex_index * 3 + 2]);

            float3 n0, n1, n2;

            if (it0.normal_index == -1 || it1.normal_index == -1 | it2.normal_index == -1)
            {
                float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
                n2 = n1 = n0 = normalize(cross(edge1, edge2));
            }
            else {
                auto normals = objReader.GetAttrib().normals;
                n0 = make_float3(normals[it0.normal_index * 3 + 0], normals[it0.normal_index * 3 + 1], normals[it0.normal_index * 3 + 2]);
                n1 = make_float3(normals[it1.normal_index * 3 + 0], normals[it1.normal_index * 3 + 1], normals[it1.normal_index * 3 + 2]);
                n2 = make_float3(normals[it2.normal_index * 3 + 0], normals[it2.normal_index * 3 + 1], normals[it2.normal_index * 3 + 2]);
            }
            triangles.push_back(Triangle { v0, v1, v2, n0, n1, n2, color});
        }
    }

    BVHTree* finalize() const { return createBVH(triangles); }
};

#endif
