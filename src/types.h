#ifndef H_TYPES
#define H_TYPES
#include <limits>

#define inf 99999999

#ifdef __CUDACC__
#define HYBRID __host__ __device__
#else
#define HYBRID
#endif 

// Asserts that the current location is within the space
#define CUDA_LIMIT(x,y) { if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return; }

__device__ uint dim1(uint x, uint y) { return x + y * WINDOW_WIDTH; }

// I would love to implement this as a union with a float3 as base member but CUDA
// goes all haywire, so hopefully the compilers remove back and forth coverting.
struct vec3 {
    float x;
    float y;
    float z;

    HYBRID vec3() : x(0), y(0), z(0) {}
    HYBRID vec3(float v) : x(v), y(v), z(v) {}
    HYBRID vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    HYBRID vec3(float3 e) : x(e.x), y(e.y), z(e.z) {}
    HYBRID float3 tof3() const { return make_float3(x,y,z); }
};

HYBRID inline float dot(const vec3& lhs, const vec3& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
HYBRID inline vec3 operator - (const vec3& lhs, const vec3& rhs) { return vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); } 
HYBRID inline vec3 operator + (const vec3& lhs, const vec3& rhs) { return vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); } 
HYBRID inline vec3 operator * (const vec3& lhs, const vec3& rhs) { return vec3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); } 
HYBRID inline vec3 operator * (const float s, const vec3& rhs) { return vec3(s * rhs.x, s * rhs.y, s * rhs.z); } 
HYBRID inline vec3 normalized (const vec3 &v) { return rsqrt(dot(v,v))*v; }


struct Sphere
{
    vec3 pos;
    float radius;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

__device__ Ray getRayForPixel(unsigned int x, unsigned int y)
{
    float xf = 2 * (x / (float)WINDOW_WIDTH) - 1;
    float yf = 2 * (y / (float)WINDOW_HEIGHT) - 1;
    float ar = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;

    float camDis = 1.0;
    vec3 pixel(xf * ar, camDis, yf);
    vec3 eye(0);
    return Ray {
        eye,
        normalized(pixel - eye)
    };
}

__device__ bool raySphereIntersect(const Sphere& sphere, const Ray& ray, float* dis, vec3* normal)
{
    vec3 c = sphere.pos - ray.origin;
    float t = dot(c, ray.direction);
    vec3 q = c - (t * ray.direction);
    float p2 = dot(q,q);
    if (p2 > sphere.radius * sphere.radius) return false;
    t -= sqrt(sphere.radius * sphere.radius - p2);

    vec3 pos = ray.origin + t * ray.direction;
    *normal = normalized(pos - sphere.pos);
    *dis = t;
    return t > 0;
}
#endif
