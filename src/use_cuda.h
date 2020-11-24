#ifndef H_USE_CUDA
#define H_USE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <surface_functions.h>
#include <cuda_surface_types.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <texture_types.h>
#include <curand_kernel.h>
#include "cutil_math.h"

#include "constants.h"

#ifdef __CUDACC__
#define HYBRID __host__ __device__
#else
#define HYBRID
#endif 

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

HYBRID inline uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

HYBRID inline float rand(uint& seed)
{
    seed = wang_hash(seed);

    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    seed &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    seed |= ieeeOne;                          // Add fractional part to 1.0

    float  f = reinterpret_cast<float&>(seed);       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

HYBRID inline uint getSeed(uint x, uint y, float time)
{
    return (x + WINDOW_WIDTH * y) * (uint)(time * 100);
}

#endif
