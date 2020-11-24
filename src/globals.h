#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global pointer to the bvh buffer
__constant__ BVHNode* DBVH;

// global pointer to triangle buffer, one for just vertices
__constant__ TriangleV* DTriangles;

// the rest in a different buffer
__constant__ TriangleD* DTriangleData;

__constant__ Material* DMaterials;

__constant__ Sphere DLight;

__constant__ float3 DLight_Color;

__device__ AtomicQueue<Ray> DRayQueue;

__device__ AtomicQueue<Ray> DShadowRayQueue;

__device__ AtomicQueue<Ray> DRayQueueNew;

__device__ DSizedBuffer<Sphere> DSpheres;

static TriangleV* HTriangles;
static TriangleD* HTriangleData;
static Material* HMaterials;
static HSizedBuffer<Sphere> HSpheres;
static BVHNode* HBVH;

#ifdef __CUDA_ARCH__
#define _GTriangles DTriangles
#define _GTriangleData DTriangleData
#define _GSpheres DSpheres
#define _GMaterials DMaterials
#define _GBVH DBVH
#else
#define _GTriangles HTriangles
#define _GTriangleData HTriangleData
#define _GSpheres HSpheres
#define _GMaterials HMaterials
#define _GBVH HBVH
#endif

#endif
