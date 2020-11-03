#ifndef H_TYPES
#define H_TYPES

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

};

struct Ray
{
    float3 origin;
    float3 direction;
};

struct Isect
{
    const Sphere* collider;
    float t;
    float3 normal;
    float3 pos;
};

#endif
