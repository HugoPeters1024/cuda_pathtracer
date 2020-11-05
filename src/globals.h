#ifndef H_GLOBALS
#define H_GLOBALS

#include "use_cuda.h"
#include "types.h"

// global pointer to the bvh buffer
__device__ BVHNode* GBVH;

// global pointer to triangle buffer
__device__ Triangle* GTriangles;

#endif
