#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global NEE switch
__device__ __constant__ bool DNEE;

__device__ __constant__ TriangleV* DVertices;

__device__ __constant__ TriangleD* DVertexData;

__device__ __constant__ Model* DModels;

__device__ __constant__ Instance* DInstances;

__device__ __constant__ TopLevelBVH* DTopBVH;

__device__ __constant__ Material* DMaterials;

__device__ AtomicQueue<RayPacked> DRayQueue;

__device__ AtomicQueue<RayPacked> DShadowRayQueue;

__device__ AtomicQueue<RayPacked> DRayQueueNew;

__device__ __constant__ DSizedBuffer<Sphere> DSpheres;

__device__ __constant__ DSizedBuffer<Plane> DPlanes;

__device__ __constant__ DSizedBuffer<SphereLight> DSphereLights;

static bool HNEE;
static TriangleV* HVertices;
static TriangleD* HVertexData;
static Model* HModels;
static Instance* HInstances;
static TopLevelBVH* HTopBVH;
static Material* HMaterials;
static HSizedBuffer<Sphere> HSpheres;
static HSizedBuffer<Plane> HPlanes;
static HSizedBuffer<SphereLight> HSphereLights;

#ifdef __CUDA_ARCH__
#define _NEE DNEE
#define _GVertices DVertices
#define _GVertexData DVertexData
#define _GModels DModels
#define _GInstances DInstances
#define _GSpheres DSpheres
#define _GPlanes DPlanes
#define _GMaterials DMaterials
#define _GTopBVH DTopBVH
#define _GSphereLights DSphereLights
#else
#define _NEE HNEE
#define _GVertices HVertices
#define _GVertexData HVertexData
#define _GModels HModels
#define _GInstances HInstances
#define _GSpheres HSpheres
#define _GPlanes HPlanes
#define _GMaterials HMaterials
#define _GBVH HBVH
#define _GTopBVH HTopBVH
#define _GSphereLights HSphereLights
#endif

#define NEE

#endif
