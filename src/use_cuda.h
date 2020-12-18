#ifndef H_USE_CUDA
#define H_USE_CUDA

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
#include <cuda_texture_types.h>
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

HYBRID inline uint rand_xorshift(uint seed)
{
    // Xorshift algorithm from George Marsaglia's paper
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return seed;
}

HYBRID inline float rand(uint& seed)
{
    seed = rand_xorshift(seed);
    // Faster on cuda probably
    return seed * 2.3283064365387e-10f;

    /*
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    seed &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    seed |= ieeeOne;                          // Add fractional part to 1.0

    float  f = reinterpret_cast<float&>(seed);       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
    */
}

HYBRID inline uint getSeed(uint x, uint y, float time)
{
    return wang_hash((x + WINDOW_WIDTH * y) * (uint)(time * 100));
}

HYBRID inline float at(const float3& v, uint i)
{
    return ((const float*)&v)[i];
}

inline cudaTextureObject_t loadTexture(const char* filename)
{
  int width, height, nrChannels;

  unsigned char* data3 = stbi_load(filename, &width, &height, &nrChannels, 0);
  if (!data3) {
    fprintf(stderr, "Could not load texture: %s", filename);
    exit(8);
  } else { printf("Loaded texture %s (%ix%i)\n", filename, width, height);
  }

  assert(nrChannels >= 3);

  // Convert the float data to 4 component float
  float* fdata = (float*)malloc(width*height*4*sizeof(float));
  float r = 1.0f / 255.0f;

  for(int y=0; y<height; y++)
  {
    for(int x=0; x<width; x++) {
      fdata[x*4+(height-y-1)*4*width+0] = data3[x*nrChannels+y*nrChannels*width+0]*r;
      fdata[x*4+(height-y-1)*4*width+1] = data3[x*nrChannels+y*nrChannels*width+1]*r;
      fdata[x*4+(height-y-1)*4*width+2] = data3[x*nrChannels+y*nrChannels*width+2]*r;
      fdata[x*4+(height-y-1)*4*width+3] = 1;
    }
  }

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray* cuArray;
  cudaSafe( cudaMallocArray(&cuArray, &channelDesc, width, height) );
  cudaSafe( cudaMemcpyToArray(cuArray, 0, 0, fdata, width*height*sizeof(float4), cudaMemcpyHostToDevice));

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.normalizedCoords = true;
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;

  cudaResourceViewDesc viewDesc;
  memset(&viewDesc, 0, sizeof(viewDesc));
  viewDesc.format = cudaResViewFormatFloat4;
  viewDesc.width = width * sizeof(float4);

  cudaTextureObject_t ret = 0;
  cudaSafe(cudaCreateTextureObject(&ret, &resDesc, &texDesc, nullptr));

  stbi_image_free(data3);
  free(fdata);
  return ret;
}

inline cudaTextureObject_t loadTextureHDR(const char* filename)
{
  int width, height, nrChannels;

  float* data3 = stbi_loadf(filename, &width, &height, &nrChannels, 0);
  if (!data3) {
    fprintf(stderr, "Could not load texture: %s", filename);
    exit(8);
  } else { printf("Loaded texture %s (%ix%i)\n", filename, width, height);
  }

  assert(nrChannels >= 3);

  // Convert the float data to 4 component float
  float* fdata = (float*)malloc(width*height*4*sizeof(float));

  for(int y=0; y<height; y++)
  {
    for(int x=0; x<width; x++) {
      fdata[x*4+(height-y-1)*4*width+0] = data3[x*nrChannels+y*nrChannels*width+0];
      fdata[x*4+(height-y-1)*4*width+1] = data3[x*nrChannels+y*nrChannels*width+1];
      fdata[x*4+(height-y-1)*4*width+2] = data3[x*nrChannels+y*nrChannels*width+2];
      fdata[x*4+(height-y-1)*4*width+3] = 1;
    }
  }

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray* cuArray;
  cudaSafe( cudaMallocArray(&cuArray, &channelDesc, width, height) );
  cudaSafe( cudaMemcpyToArray(cuArray, 0, 0, fdata, width*height*sizeof(float4), cudaMemcpyHostToDevice));

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.normalizedCoords = true;
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;

  cudaResourceViewDesc viewDesc;
  memset(&viewDesc, 0, sizeof(viewDesc));
  viewDesc.format = cudaResViewFormatFloat4;
  viewDesc.width = width * sizeof(float4);

  cudaTextureObject_t ret = 0;
  cudaSafe(cudaCreateTextureObject(&ret, &resDesc, &texDesc, nullptr));

  stbi_image_free(data3);
  free(fdata);
  return ret;
}

#endif
