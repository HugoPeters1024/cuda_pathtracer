#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global NEE switch
__device__ __constant__ bool DNEE;

__device__ AtomicQueue<RayPacked> DRayQueue;

__device__ AtomicQueue<RayPacked> DShadowRayQueue;

__device__ AtomicQueue<RayPacked> DRayQueueNew;

__device__ __constant__ DSizedBuffer<TriangleLight> DTriangleLights;

static bool HNEE;

#ifdef __CUDA_ARCH__
#define _NEE (DNEE && DTriangleLights.size > 0)
//#define _GVertices DVertices
//#define _GVertexData DVertexData
//#define _GModels DModels
//#define _GInstances DInstances
//#define _GSpheres DSpheres
//#define _GPlanes DPlanes
//#define _GMaterials DMaterials
//#define _GTopBVH DTopBVH
#else
#define _NEE HNEE
//#define _GVertices HVertices
//#define _GVertexData HVertexData
//#define _GModels HModels
//#define _GInstances HInstances
//#define _GSpheres HSpheres
//#define _GPlanes HPlanes
//#define _GMaterials HMaterials
//#define _GBVH HBVH
//#define _GTopBVH HTopBVH
#endif

#define NEE

#endif
