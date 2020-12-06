#ifndef H_TYPES
#define H_TYPES

#include <vector>
#include <algorithm>
#include <stack>

#define GL_GLEXT_PROTOTYPES 1
#define GL3_PROTOTYPES      1
#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "use_cuda.h"
#include <limits>
#include <vector_functions.h>

#include "constants.h"
#include "vec.h"

#define inf 99999999

// Asserts that the current location is within the space
#define CUDA_LIMIT(x,y) { if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return; }

typedef uint MATERIAL_ID;
struct Material
{
    float3 diffuse_color;
    float3 specular_color;
    float reflect;
    float glossy;
    float transmit;
    float refractive_index;
    float3 absorption;
    cudaTextureObject_t texture;
    cudaTextureObject_t normal_texture;
    bool hasTexture = false;
    bool hasNormalMap = false;

    Material() {}
    Material(float3 color, float reflect, float glossy, float transmit, float refractive_index, float3 specular_color, float3 absorption)
        : diffuse_color(color), reflect(reflect), glossy(glossy), 
          transmit(transmit), refractive_index(refractive_index),
          specular_color(specular_color), absorption(absorption) {}

    static Material DIFFUSE(float3 color) { return Material(color, 0, 0, 0, 0, make_float3(0), make_float3(0)); }
};

struct Sphere
{
    float3 pos;
    float radius;
    MATERIAL_ID material;

    Sphere() {}
    Sphere(float3 pos, float radius, MATERIAL_ID material) 
        : pos(pos), radius(radius), material(material) {}

    HYBRID inline float3 centroid() { return pos; }
};

struct SphereLight : public Sphere
{
    float3 color;
    SphereLight(float3 pos, float radius, float3 color)
        : Sphere(pos, radius, 0), color(color) {}
};

struct Plane
{
    float3 normal;
    float d;
    MATERIAL_ID material;

    Plane() {}
    Plane(float3 normal, float d, MATERIAL_ID material)
        : normal(normal), d(d), material(material) {}
};

struct PointLight
{
    float3 pos;
    float3 color;

    PointLight(float3 pos, float3 color) : pos(pos), color(color) {}
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
    float length;
    uint pixeli;

    HYBRID Ray() {}
    HYBRID Ray(float3 origin, float3 direction, uint pixeli) : origin(origin), direction(direction), pixeli(pixeli) { length = 9999999; }
    HYBRID Ray(float3 origin, float3 direction, uint px, uint py) : origin(origin), direction(direction), pixeli(px + py * WINDOW_WIDTH) { length = 9999999;}

    __device__ float getSortingKey() const { return (float)pixeli; }
};

enum PRIMITIVE_TYPE { TRIANGLE, SPHERE, PLANE, LIGHT };

struct __align__(16) HitInfo
{
    PRIMITIVE_TYPE primitive_type;
    bool intersected;
    uint primitive_id;
    float t;
};

struct __align__(16) Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;
    float3 n0;
    float3 n1;
    float3 n2;
    float2 uv0;
    float2 uv1;
    float2 uv2;
    MATERIAL_ID material;
    Matrix4 TBN;

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

struct __align__(16) TriangleV
{
    float3 v0, v1, v2;
    TriangleV(float3 v0, float3 v1, float3 v2) : v0(v0), v1(v1), v2(v2) {}
};

struct __align__(16) TriangleD
{
    float3 n0, n1, n2;
    float2 uv0, uv1, uv2;
    MATERIAL_ID material;
    Matrix4 TBN;

    TriangleD(float3 n0, float3 n1, float3 n2, float2 uv0, float2 uv1, float2 uv2, MATERIAL_ID material, Matrix4 TBN)
        : n0(n0), n1(n1), n2(n2), uv0(uv0), uv1(uv1), uv2(uv2), material(material), TBN(TBN) {}
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

    ~BVHTree()
    {
        delete child1;
        delete child2;
    }
};


struct __align__(16) BVHNode
{
    Box boundingBox;
    // either a leaf or a child
    union {
        uint child1;
        uint t_start;
    };
    uint t_data;
    HYBRID uint split_plane() const { return t_data >> 30; }
    HYBRID uint t_count() const { return t_data & (0xffffffff>>2); }
    HYBRID bool isLeaf() const { return t_count() > 0; }

    static BVHNode MakeChild(Box boundingBox, uint t_start, uint t_count) 
    {
        BVHNode ret;
        ret.boundingBox = boundingBox;
        ret.t_start = t_start;
        ret.t_data = t_count;
        return ret;
    }
    static BVHNode MakeNode(Box boundingBox, uint child1, uint split_plane) 
    {
        BVHNode ret;
        ret.boundingBox = boundingBox;
        ret.child1 = child1;
        ret.t_data = split_plane << 30;
        return ret;
    }
};

struct __align__(16) TraceState
{
    float3 mask;
    float3 accucolor;
    float3 light;
    bool fromSpecular;
};

struct TraceStateSOA
{
    float4* masks;
    float4* accucolors;
    float4* lights;

    __device__ TraceState getState(uint i) const
    {
        float4 mask = masks[i];
        float4 accucolor = accucolors[i];
        float4 light = lights[i];

        // For some reason this is much much faster than a constructor...
        return TraceState
        {
            get3f(mask),
            get3f(accucolor),
            get3f(light),
            __float_as_int(mask.w),
        };
    }

    __device__ void setState(uint i, const TraceState& state)
    {
        masks[i] = make_float4(state.mask, __int_as_float(state.fromSpecular));
        accucolors[i] = make_float4(state.accucolor, 0);
        lights[i] = make_float4(state.light, 0);
    }
};


template <class T>
struct AtomicQueue
{
    T* values;
    uint size;

    AtomicQueue() {}
    AtomicQueue(uint capacity)
    {
        cudaSafe( cudaMalloc(&values, capacity * sizeof(T)) );
        size = 0;
    }

    __device__ T const& operator[](int index) { return values[index]; }

    void syncFromDevice(const AtomicQueue<T>& origin)
    {
        cudaSafe (cudaMemcpyFromSymbol(this, origin, sizeof(AtomicQueue<T>)) );
    }

    void syncToDevice(const AtomicQueue<T>& destination)
    {
        cudaSafe( cudaMemcpyToSymbol(destination, this, sizeof(AtomicQueue<T>)) );
    }

    __device__ inline void push(const T& ray)
    {
        values[atomicAdd(&size, 1)] = ray;
    }

    inline void clear() { size = 0; }
};

template <class T>
struct DSizedBuffer
{
    T* values;
    uint size;
    const DSizedBuffer<T>* dest;

    DSizedBuffer() {}
    DSizedBuffer(T* src, int size, const DSizedBuffer<T>* dest) : size(size), dest(dest)
    {
        cudaSafe( cudaMalloc(&values, size * sizeof(T)));
        cudaSafe( cudaMemcpy(values, src, size * sizeof(T), cudaMemcpyHostToDevice) );
        cudaSafe( cudaMemcpyToSymbol(*dest, this, sizeof(DSizedBuffer<T>)) );
    }

    void update(T* newsrc)
    {
        cudaSafe( cudaMemcpy(values, newsrc, size * sizeof(T), cudaMemcpyHostToDevice) );
    }

    __device__ T const& operator[](int index) { return values[index]; }
};

template <class T>
struct HSizedBuffer
{
    T* values;
    uint size;

    HSizedBuffer() {}
    HSizedBuffer(T* src, int size) : values(src), size(size)
    {
    }

    T const& operator[](int index) { return values[index]; }
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

        float speed = 0.08f;
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

    HYBRID inline Ray getRay(unsigned int x, unsigned int y, uint& seed) const {
        float xf = ((float)x + rand(seed)) / WINDOW_WIDTH;
        float yf = ((float)y + rand(seed)) / WINDOW_HEIGHT;
        float3 point = distort(lt + xf * u + yf * v);

        float3 direction = normalize(point - eye);
        return Ray(eye, direction, x,y);
    }

    HYBRID inline Ray getRay(unsigned int x, unsigned int y) const {
        float xf = ((float)x) / WINDOW_WIDTH;
        float yf = ((float)y) / WINDOW_HEIGHT;
        float3 point = distort(lt + xf * u + yf * v);

        float3 direction = normalize(point - eye);
        return Ray(eye, direction, x,y);
    }

    HYBRID float3 distort(const float3& p) const
    {
        float3 center = eye + d * viewDir;
        float3 fromCenter = p - center;
        float r = length(p - center);
        float rd = r + 0.2 * r * r * r;
        return center + fromCenter * (rd/r);
    }
};


#endif
