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
#include <vector_types.h>

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif
