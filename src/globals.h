#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global pointer to the bvh buffer
__constant__ BVHNode* GBVH;

// global pointer to triangle buffer, one for just vertices
__constant__ TriangleV* GTriangles;

// the rest in a different buffer
__constant__ TriangleD* GTriangleData;

__constant__ Sphere GLight;

// the global application time
__constant__ float GTime;

__device__ AtomicQueue<Ray> GRayQueue;

__device__ AtomicQueue<Ray> GShadowRayQueue;

__device__ AtomicQueue<Ray> GRayQueueNew;

__device__ SizedBuffer<Sphere> GSpheres;

#endif
