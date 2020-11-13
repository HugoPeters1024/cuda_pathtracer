#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global pointer to the bvh buffer
__constant__ BVHNode* GBVH;

// global pointer to triangle buffer
__constant__ Triangle* GTriangles;

__constant__ Sphere GLight;

// the global application time
__constant__ float GTime;

__device__ AtomicQueue<Ray> GRayQueue;

__device__ AtomicQueue<Ray> GShadowRayQueue;

__device__ AtomicQueue<Ray> GRayQueueNew;

#endif
