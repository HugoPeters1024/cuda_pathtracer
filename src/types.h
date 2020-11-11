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

#include "constants.h"

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
    inline float getSurfaceArea() const
    {
        float3 minToMax = vmax - vmin;
        float lx = minToMax.x;
        float ly = minToMax.y;
        float lz = minToMax.z;
        return 2*lx*ly + 2*lx*lz + 2*ly*lz;
    }

    static Box fromPoint(const float3 p)
    {
        return Box {
            p,p
        };
    }

    void consumePoint(const float3& p)
    {
        vmin = fminf(p, vmin);
        vmax = fmaxf(p, vmax);
    }

    bool overlaps(const Box& other) const {
        return vmax.x >= other.vmin.x && other.vmax.x >= vmin.x &&
               vmax.y >= other.vmin.y && other.vmax.y >= vmin.y &&
               vmax.z >= other.vmin.z && other.vmax.z >= vmin.z;
    }
};

struct __align__(16) Ray
{
    float3 origin;
    float3 direction;
    float3 invdir;
    int signs[3];
    float3 shadowTarget;
    bool active;
};

HYBRID Ray makeRay(float3 origin, float3 direction)
{
    Ray ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.invdir = 1.0 / ray.direction;
    ray.signs[0] = (int)(ray.invdir.x < 0);
    ray.signs[1] = (int)(ray.invdir.y < 0);
    ray.signs[2] = (int)(ray.invdir.z < 0);
    return ray;
}

struct HitInfo
{
    bool intersected;
    uint triangle_id;
    float t;
    float3 normal;
};

struct __align__(16) Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;
    float3 n0;
    float3 n1;
    float3 n2;
    float3 color;
    float reflect;

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

    float getSurfaceArea() const
    {
        return 0.5 * (v0.x*(v1.y - v2.y) + v1.x*(v2.y - v0.y) + v2.x*(v0.y-v1.y));
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

struct BVHSplittingTree
{
    float cost;
    BVHSplittingTree* child1;
    BVHSplittingTree* child2;
};

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

struct __align__(16) BVHNode
{
    Box boundingBox;
    uint child2;
    uint split_plane;
    uint t_start;
    uint t_count;

    HYBRID bool isLeaf() const { return t_count > 0; }
};

class Camera
{
private:
    bool has_moved;
    void recalculate() {
        float3 center = eye + d * viewDir;
        u = normalize(cross(make_float3(0,1,0), viewDir));
        v = normalize(cross(viewDir, u));

        float ar = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;
        lt = center - u * ar - v;

        u = 2 * ar * u;
        v = 2 * v;
    }

public:
    float3 eye, viewDir;
    float3 lt, u, v;
    float d;

    Camera(float3 eye, float3 viewDir, float d) : eye(eye), viewDir(viewDir), d(d) {}

    void update(GLFWwindow* window) {
        has_moved = false;
        float3 oldEye = eye;
        float3 oldViewDir = viewDir;

        float speed = 0.03f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) eye += speed * viewDir;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) eye -= speed * viewDir;
        float3 side = normalize(cross(make_float3(0,1,0), viewDir));
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) eye -= speed * side;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) eye += speed * side;

        // Look changes
        float look_speed = 0.02;
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) viewDir.y += look_speed;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) viewDir.y -= look_speed;
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) viewDir -= look_speed * side;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) viewDir += look_speed * side;
        viewDir = normalize(viewDir);
        recalculate();
        has_moved = (oldEye != eye || oldViewDir != viewDir);
    }

    inline bool hasMoved() const { return has_moved; }

    HYBRID inline Ray getRay(unsigned int x, unsigned int y) const {
        float xf = x / (float)WINDOW_WIDTH;
        float yf = y / (float)WINDOW_HEIGHT;
        float3 point = lt + xf * u + yf * v;

        float3 direction = normalize(point - eye);
        return makeRay(eye, direction);
    }
};


#endif
