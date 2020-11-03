#ifndef H_TYPES
#define H_TYPES
#include <limits>

#define inf 99999999

#ifdef __CUDACC__
#define CUDA __host__ __device__
#else
#define CUDA
#endif 

// Asserts that the current location is within the space
#define CUDA_LIMIT(x,y) { if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return; }

__device__ uint dim1(uint x, uint y) { return x + y * WINDOW_WIDTH; }


struct Vector3f {
    float x;
    float y;
    float z;

    CUDA Vector3f() : x(0), y(0), z(0) {}
    CUDA Vector3f(float v) : x(v), y(v), z(v) {}
    CUDA Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
};

CUDA inline float dot(const Vector3f& lhs, const Vector3f& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
CUDA inline Vector3f operator - (const Vector3f& lhs, const Vector3f& rhs) { return Vector3f(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); } 
CUDA inline Vector3f operator + (const Vector3f& lhs, const Vector3f& rhs) { return Vector3f(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); } 
CUDA inline Vector3f operator * (const Vector3f& lhs, const Vector3f& rhs) { return Vector3f(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); } 
CUDA inline Vector3f operator * (const float s, const Vector3f& rhs) { return Vector3f(s * rhs.x, s * rhs.y, s * rhs.z); } 
CUDA Vector3f normalized(const Vector3f &v) { float l = sqrt(dot(v,v)); return Vector3f(v.x/l, v.y/l, v.z/l); }


struct Sphere
{
    Vector3f pos;
    float radius;
};

struct Ray
{
    Vector3f origin;
    Vector3f direction;
};

__device__ bool raySphereIntersect(const Sphere& sphere, const Ray& ray, float* dis, Vector3f* normal)
{
    Vector3f c = sphere.pos - ray.origin;
    float t = dot(c, ray.direction);
    Vector3f q = c - (t * ray.direction);
    float p2 = dot(q,q);
    if (p2 > sphere.radius * sphere.radius) return false;
    t -= sqrt(sphere.radius * sphere.radius - p2);

    Vector3f pos = ray.origin + t * ray.direction;
    *normal = normalized(pos - sphere.pos);
    *dis = t;
    return t > 0;
}
#endif
