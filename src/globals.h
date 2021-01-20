#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global NEE switch
__device__ __constant__ bool DNEE;
__device__ __constant__ bool DCACHE;

__device__ AtomicQueue<RayPacked> DRayQueue;

__device__ AtomicQueue<RayPacked> DShadowRayQueue;

__device__ AtomicQueue<RayPacked> DRayQueueNew;

__device__ __constant__ DSizedBuffer<TriangleLight> DTriangleLights;

static bool HNEE;
static bool HCACHE;

#ifdef __CUDA_ARCH__
#define _NEE (DNEE && DTriangleLights.size > 0)
#define _CACHE DDCACHE
#else
#define _NEE HNEE
#define _CACHE HCACHE
#endif

#define NEE

#endif
