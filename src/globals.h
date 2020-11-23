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

__constant__ Material* GMaterials;

__constant__ Sphere GLight;

__constant__ float3 GLight_Color;

__device__ AtomicQueue<Ray> GRayQueue;

__device__ AtomicQueue<Ray> GShadowRayQueue;

__device__ AtomicQueue<Ray> GRayQueueNew;

__device__ SizedBuffer<Sphere> GSpheres;

#endif
