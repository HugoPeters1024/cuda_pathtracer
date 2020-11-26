#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global pointer to the bvh buffer
__device__ __constant__ BVHNode* DBVH;

// global pointer to triangle buffer, one for just vertices
__device__ __constant__ TriangleV* DTriangles;

// the rest in a different buffer
__device__ __constant__ TriangleD* DTriangleData;

__device__ __constant__ Material* DMaterials;

__device__ __constant__ Sphere DLight;

__device__ __constant__ float3 DLight_Color;

__device__ AtomicQueue<Ray> DRayQueue;

__device__ AtomicQueue<Ray> DShadowRayQueue;

__device__ AtomicQueue<Ray> DRayQueueNew;

__device__ __constant__ DSizedBuffer<Sphere> DSpheres;

__device__ __constant__ DSizedBuffer<Plane> DPlanes;

static TriangleV* HTriangles;
static TriangleD* HTriangleData;
static Material* HMaterials;
static HSizedBuffer<Sphere> HSpheres;
static HSizedBuffer<Plane> HPlanes;
static BVHNode* HBVH;

#ifdef __CUDA_ARCH__
#define _GTriangles DTriangles
#define _GTriangleData DTriangleData
#define _GSpheres DSpheres
#define _GPlanes DPlanes
#define _GMaterials DMaterials
#define _GBVH DBVH
#else
#define _GTriangles HTriangles
#define _GTriangleData HTriangleData
#define _GSpheres HSpheres
#define _GPlanes HPlanes
#define _GMaterials HMaterials
#define _GBVH HBVH
#endif

//#define NEE

#endif
