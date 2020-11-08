#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global pointer to the bvh buffer
__constant__ BVHNode* GBVH;

// global pointer to triangle buffer
__constant__ Triangle* GTriangles;

// the global application time
__device__ float GTime;

#endif
