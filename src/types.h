#ifndef H_TYPES
#define H_TYPES

#include "use_cuda.h"
#include <limits>

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
};

struct Ray
{
    float3 origin;
    float3 direction;
};

__device__ Ray getRayForPixel(unsigned int x, unsigned int y)
{
    float xf = 2 * (x / (float)WINDOW_WIDTH) - 1;
    float yf = 2 * (y / (float)WINDOW_HEIGHT) - 1;
    float ar = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;

    float camDis = 1.0;
    float3 pixel = make_float3(xf * ar, camDis, yf);
    float3 eye = make_float3(0);
    return Ray {
        eye,
        normalize(pixel - eye)
    };
}

__device__ bool raySphereIntersect(const Sphere& sphere, const Ray& ray, float* dis, float3* pos, float3* normal)
{
    float3 c = sphere.pos - ray.origin;
    float t = dot(c, ray.direction);
    float3 q = c - (t * ray.direction);
    float p2 = dot(q,q);
    if (p2 > sphere.radius * sphere.radius) return false;
    t -= sqrtf(sphere.radius * sphere.radius - p2);

    *pos = ray.origin + t * ray.direction;
    *normal = normalize(*pos - sphere.pos);
    *dis = t;
    return t > 0;
}
#endif
