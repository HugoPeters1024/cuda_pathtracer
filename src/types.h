#ifndef H_TYPES
#define H_TYPES

#include <vector>
#include <algorithm>
#include <stack>
#include <functional>

#define GL_GLEXT_PROTOTYPES 1
#define GL3_PROTOTYPES      1
#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "use_cuda.h"
#include <limits>
#include <vector_functions.h>
#include <emmintrin.h>

#include "constants.h"
#include "vec.h"
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#define inf 99999999

// Asserts that the current location is within the space
#define CUDA_LIMIT(x,y) { if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return; }

typedef uint MATERIAL_ID;
struct Material
{
    float3 diffuse_color;
    float3 specular_color;
    float3 emission;
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
    Material(float3 color, float reflect, float glossy, float transmit, float refractive_index, float3 specular_color, float3 absorption, float3 emission)
        : diffuse_color(color), reflect(reflect), glossy(glossy), 
          transmit(transmit), refractive_index(refractive_index),
          specular_color(specular_color), absorption(absorption),
          emission(emission) {}

    static Material DIFFUSE(float3 color) { return Material(color, 0, 0, 0, 0, make_float3(0), make_float3(0), make_float3(0) ); }
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
    SphereLight() {}
    SphereLight(float3 pos, float radius, float3 color)
        : Sphere(pos, radius, 0), color(color) {}
};

struct __align__(16) TriangleV
{
    float3 v0, v1, v2;
    HYBRID TriangleV(float3 v0, float3 v1, float3 v2) : v0(v0), v1(v1), v2(v2) {}
};

struct __align__(16) TriangleD
{
    float3 normal, tangent, bitangent;
    float2 uv0, uv1, uv2;
    MATERIAL_ID material;

    TriangleD() {}


    TriangleD(float3 normal, float3 tangent, float3 bitangent, float2 uv0, float2 uv1, float2 uv2, MATERIAL_ID material)
        : normal(normal), tangent(tangent), bitangent(bitangent), uv0(uv0), uv1(uv1), uv2(uv2), material(material) {}
};

struct __align__(16) RadianceCache
{
    float radianceCache[8];
    float additionCache[8];
    float additionCacheCount[8];
    float radianceTotal;
};

struct TriangleLight
{
    uint triangle_index;
    uint instance_index;
};

static float3* SORTING_SOURCE;

static bool __compare_triangles_x (uint a, uint b) {
    return SORTING_SOURCE[a].x < SORTING_SOURCE[b].x;
}
static bool __compare_triangles_y (uint a, uint b) {
    return SORTING_SOURCE[a].y < SORTING_SOURCE[b].y;
}
static bool __compare_triangles_z (uint a, uint b) {
    return SORTING_SOURCE[a].z < SORTING_SOURCE[b].z;
}

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

    PointLight() {}
    PointLight(float3 pos, float3 color) : pos(pos), color(color) {}
};

struct Box
{
    float3 vmin;
    float3 vmax;

    HYBRID Box() {}
    HYBRID Box(float3 vmin, float3 vmax) : vmin(vmin), vmax(vmax) {}
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

    static Box insideOut()
    {
        return Box {
            make_float3(9999999),
            make_float3(-9999999)
        };
    }
    
    static Box merged(const Box& a, const Box& b)
    {
        Box ret = Box::insideOut();
        ret.consumeBox(a);
        ret.consumeBox(b);
        return ret;
    }

    void consumeBox(const Box& a)
    {
        vmin = fminf(vmin, a.vmin);
        vmax = fmaxf(vmax, a.vmax);
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

struct SSEBox
{
    __m128 vmin, vmax;

    static SSEBox insideOut()
    {
        return SSEBox {
            _mm_setr_ps(999999, 99999, 99999, 0),
            _mm_setr_ps(-999999, -99999, -99999, 0),
        };
    }

    static SSEBox fromTriangle(const TriangleV& t)
    {
        __m128 sse_v0 = _mm_setr_ps(t.v0.x, t.v0.y, t.v0.z, 0);
        __m128 sse_v1 = _mm_setr_ps(t.v1.x, t.v1.y, t.v1.z, 0);
        __m128 sse_v2 = _mm_setr_ps(t.v2.x, t.v2.y, t.v2.z, 0);
        return SSEBox {
            _mm_min_ps(sse_v0, _mm_min_ps(sse_v1, sse_v2)),
            _mm_max_ps(sse_v0, _mm_max_ps(sse_v1, sse_v2)),
        };
    }

    inline void consumeBox(const SSEBox& a)
    {
        vmin = _mm_min_ps(vmin, a.vmin);
        vmax = _mm_max_ps(vmax, a.vmax);
    }

    inline void consumePoint(const __m128& sse_p)
    {
        vmin = _mm_min_ps(vmin, sse_p);
        vmax = _mm_max_ps(vmax, sse_p);
    }

    inline Box toNormalBox()
    {
        return Box {
            make_float3(vmin[0], vmin[1], vmin[2]),
            make_float3(vmax[0], vmax[1], vmax[2]),
        };
    }

    inline float getSurfaceArea() const
    {
        __m128 minToMax = _mm_sub_ps(vmax, vmin);
        float lx = minToMax[0];
        float ly = minToMax[1];
        float lz = minToMax[2];
        return 2*lx*ly + 2*lx*lz + 2*ly*lz;
    }
};


struct Ray
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

struct __align__(16) RayPacked
{
    float4 origin;
    float4 direction;

    HYBRID inline Ray getRay() const
    {
        Ray ret;
        ret.origin = get3f(origin);
        ret.direction = get3f(direction);
        ret.length = origin.w;
        ret.pixeli = reinterpret_cast<const uint&>(direction.w);
        return ret;
    }

    HYBRID RayPacked() {}
    HYBRID RayPacked(const Ray& ray)
    {
        origin = make_float4(ray.origin, ray.length);
        direction = make_float4(ray.direction, reinterpret_cast<const float&>(ray.pixeli));
    }
};

enum PRIMITIVE_TYPE { TRIANGLE, SPHERE, PLANE };


struct __align__(16) HitInfo
{
    PRIMITIVE_TYPE primitive_type;
    // when the primitive id is all 1's we haven't intersected
    uint primitive_id;
    uint instance_id;
    float t;

    HYBRID inline bool intersected() const { return primitive_id != 0xffffffff; }
};

struct __align__(16) HitInfoPacked
{
    float4 data;

    __device__ HitInfo getHitInfo() const
    {
        HitInfo ret = HitInfo
        {
            reinterpret_cast<const PRIMITIVE_TYPE&>(data.x),
            reinterpret_cast<const uint&>(data.y),
            reinterpret_cast<const uint&>(data.z),
            data.w
        };
        return ret;
    }

    __device__ HitInfoPacked(const HitInfo& hitInfo)
    {
        data = make_float4(
                reinterpret_cast<const float&>(hitInfo.primitive_type),
                reinterpret_cast<const float&>(hitInfo.primitive_id),
                reinterpret_cast<const float&>(hitInfo.instance_id),
                hitInfo.t);
    }
};

// three kinds of entries. sequence ends at a terminate.
enum SAMPLE_TYPE { SAMPLE_IGNORE, SAMPLE_TERMINATE, SAMPLE_BUCKET };

struct __align__(16) SampleCache
{
    SAMPLE_TYPE sample_type;
    uint triangle_id;
    int cache_bucket_id;
    float3 cum_mask;
};

struct __align__(16) BVHNode
{
    float4 vmin;
    float4 vmax;

    HYBRID inline uint t_start() const { return reinterpret_cast<const uint&>(vmin.w); }
    HYBRID inline uint child1() const { return reinterpret_cast<const uint&>(vmin.w); }

    HYBRID inline uint t_count() const { return reinterpret_cast<const uint&>(vmax.w); }
    HYBRID inline bool isLeaf() const { return t_count() > 0; }
    HYBRID inline Box getBox() const { return Box(get3f(vmin), get3f(vmax)); }

    static BVHNode MakeChild(Box boundingBox, uint t_start, uint t_count) 
    {
        BVHNode ret;
        ret.vmin = make_float4(boundingBox.vmin, reinterpret_cast<float&>(t_start));
        ret.vmax = make_float4(boundingBox.vmax, reinterpret_cast<float&>(t_count));
        return ret;
    }

    static BVHNode MakeNode(Box boundingBox, uint child1)
    {
        uint t_count = 0;
        BVHNode ret;
        ret.vmin = make_float4(boundingBox.vmin, reinterpret_cast<float&>(child1));
        ret.vmax = make_float4(boundingBox.vmax, reinterpret_cast<float&>(t_count));
        return ret;
    }
};

struct __align__(16) Model
{
    BVHNode* bvh;
    uint triangleStart;
    uint nrTriangles;
    uint nrBvhNodes;
};

struct mat4x3
{
    float4 r1;
    float4 r2;
    float4 r3;

    static mat4x3 fromGLM(const glm::mat4x4 mat)
    {
        return mat4x3
        {
            make_float4(mat[0][0], mat[1][0], mat[2][0], mat[3][0]),
            make_float4(mat[0][1], mat[1][1], mat[2][1], mat[3][1]),
            make_float4(mat[0][2], mat[1][2], mat[2][2], mat[3][2]),
        };
    }

    HYBRID float3 mul(const float3& target, const float fourth) const
    {
        const float4 v = make_float4(target, fourth);
        return make_float3(dot(v, r1), dot(v, r2), dot(v, r3));
    }
};

struct __align__(16) Instance
{
    uint model_id;
    mat4x3 transform;
    mat4x3 invTransform;
    uint material_id;
};

struct GameObject
{
    uint model_id;
    uint kind;
    uint materiald_id;
    float3 position;
    float3 rotation;
    float3 scale;

    GameObject() {}
    GameObject(uint model_id, uint material_id = 0xffffffff)
        : model_id(model_id), position(make_float3(0)), rotation(make_float3(0)), scale(make_float3(1.0f)),
          kind(0), materiald_id(material_id) {}
};

struct __align__(16) TopLevelBVH
{
    Box box;
    uint children;
    uint leaf;

    HYBRID uint child1() const { return children & 0x0000ffff; }
    HYBRID uint child2() const { return (children >> 16) & 0x0000ffff; }

    // both children 0 is impossible
    HYBRID bool isLeaf() const { return children == 0; }

static TopLevelBVH CreateLeaf(uint instanceIdx, const Box& box)
    {
        TopLevelBVH ret;
        ret.children = 0;
        ret.leaf = instanceIdx;
        ret.box = box;
        return ret;
    }

    static TopLevelBVH CreateNode(uint child1, uint child2, Box box)
    {
        TopLevelBVH ret;
        ret.children = child1 | (child2 << 16);
        ret.box = box;
        return ret;
    }
};

struct __align__(16) TraceState
{
    float3 mask;
    float3 accucolor;
    float3 light;
    bool fromSpecular;
    bool albedoSet;
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
            reinterpret_cast<bool&>(mask.w),
            reinterpret_cast<bool&>(accucolor.w),
        };
    }

    __device__ void setState(uint i, const TraceState& state)
    {
        masks[i] = make_float4(state.mask, reinterpret_cast<const float&>(state.fromSpecular));
        accucolors[i] = make_float4(state.accucolor, reinterpret_cast<const float&>(state.albedoSet));
        lights[i] = make_float4(state.light, 0);
    }
};

struct CudaTexture
{
    cudaTextureObject_t texture_id;
    int width;
    int height;
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

    __device__ inline void clear() { size = 0; }
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

struct CDF
{
    float* values;
    float* cumValues;
    uint nrItems;
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
    float focalLength;
    float aperture;

    Camera(float3 eye, float3 viewDir, float d, float focalLength, float aperture) 
        : eye(eye), viewDir(viewDir), d(d), focalLength(focalLength), aperture(aperture) {}

    void update(GLFWwindow* window) {
        has_moved = false;
        float3 oldEye = eye;
        float3 oldViewDir = viewDir;
        float oldAperture = aperture;

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

        if (glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS) aperture += 0.001;
        if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) aperture -= 0.001;
        viewDir = normalize(viewDir);
        recalculate();
        has_moved = (oldEye != eye || oldViewDir != viewDir || aperture != oldAperture);
    }

    inline bool hasMoved() const { return has_moved; }

    HYBRID inline Ray getRay(unsigned int x, unsigned int y, uint& seed) const {
        float xf = ((float)x + rand(seed)) / WINDOW_WIDTH;
        float yf = ((float)y + rand(seed)) / WINDOW_HEIGHT;
        float3 origin = distort(lt + xf * u + yf * v);
        float3 direction = origin - eye;
        const float correction = length(direction);
        direction /= correction;
        const float3 focalPoint = origin + (focalLength-d) * direction;
        const float offsetR = sqrtf(rand(seed));
        const float offsetA = rand(seed) * 2.0f * PI;
        const float2 focalOffset = make_float2(offsetR * sinf(offsetA), offsetR * cosf(offsetA));
        origin += aperture * (focalOffset.x * u + focalOffset.y * v);
        direction = normalize(focalPoint - origin);
        if (dot(direction, viewDir) < 0) direction = -direction;

        // recalculate the pixel index
        return Ray(origin - correction * direction, direction, x,y);
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
        float rd = r + 0.2f * r * r * r;
        return center + fromCenter * (rd/fmaxf(0.0001f,r));
    }
};

struct RandState
{
    uint sampleIdx;
    uint randIdx;
    float blueNoiseSample;
    cudaTextureObject_t blueNoise;
    float2 invBlueNoiseSize;
    uint seed;
};

struct SceneBuffers
{
    TriangleV* vertices;
    TriangleD* vertexData;
    RadianceCache* radianceCaches;
    uint num_triangles;
    Instance* instances;
    Model* models;
    Material* materials;
    TopLevelBVH* topBvh;
    Sphere* spheres;
    uint num_spheres;
    Plane* planes;
    uint num_planes;
};



#endif
