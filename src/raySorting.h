#ifndef H_RAYSORTING
#define H_RAYSORTING

#include "use_cuda.h"
#include "types.h"

#include "cub/device/device_radix_sort.cuh"

__global__ void kernel_generate_sorting_keys_and_indices(const Ray* rays, float* keys, uint* indices, int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = rays[i].getSortingKey();
    indices[i] = i;
}

__global__ void kernel_permute_sorting_indices(const uint* indices, const Ray* rays_in, Ray* rays_out, int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rays_out[i] = rays_in[indices[i]];
}

void sortRays(Ray** d_rays, int nrRays)
{
    // Create a cub double buffer and allocate
    cub::DoubleBuffer<float> d_keys;
    cub::DoubleBuffer<uint>  d_values;

    cudaSafe( cudaMalloc(&d_keys.d_buffers[0], nrRays * sizeof(float)) );
    cudaSafe( cudaMalloc(&d_keys.d_buffers[1], nrRays * sizeof(float)) );
    cudaSafe( cudaMalloc(&d_values.d_buffers[0], nrRays * sizeof(uint)) );
    cudaSafe( cudaMalloc(&d_values.d_buffers[1], nrRays * sizeof(uint)) );

    void *d_temp_storage = NULL;
    size_t tmp_size = 0;

    // Run once to get the temp storage values (I guess)
    cub::DeviceRadixSort::SortPairs(d_temp_storage, tmp_size, d_keys, d_values, 10);

    // Allocate the temp storage
    cudaSafe (cudaMalloc(&d_temp_storage, tmp_size));

    // Fill our keys and values
    kernel_generate_sorting_keys_and_indices<<<nrRays/1024+1, 1024>>>(*d_rays, d_keys.Current(), d_values.Current(), nrRays);

    // Sort the rays
    cub::DeviceRadixSort::SortPairs(d_temp_storage, tmp_size, d_keys, d_values, 10);

    cudaDeviceSynchronize();

    uint* keys = inspectCudaBuffer(d_values.Current(), 10);

    // Permute the given rays buffer to reflect the sort
    Ray* d_rays_result;
    cudaSafe( cudaMalloc(&d_rays_result, nrRays * sizeof(Ray)));
    kernel_permute_sorting_indices<<<nrRays/1024+1, 1024>>>(d_values.Current(), *d_rays, d_rays_result, nrRays);

    // swap the pointers and free the tmp buffers
    Ray* old = *d_rays;
    *d_rays = d_rays_result;

    cudaSafe( cudaFree(old) );
    cudaSafe( cudaFree(d_keys.d_buffers[0]) );
    cudaSafe( cudaFree(d_keys.d_buffers[1]) );
    cudaSafe( cudaFree(d_values.d_buffers[0]) );
    cudaSafe( cudaFree(d_values.d_buffers[1]) );
}


#endif
