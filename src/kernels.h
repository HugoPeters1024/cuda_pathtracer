#ifndef H_KERNELS
#define H_KERNELS

#include "use_cuda.h"
#include "types.h"


__global__ void kernel_pathtracer(cudaSurfaceObject_t texRef, Sphere* spheres, float time) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    Ray ray = getRayForPixel(x,y);
    Sphere sphere = spheres[0];
    sphere.pos.x += sin(time);


    vec3 color(0,0,0);

    float dis;
    vec3 normal;
    if (raySphereIntersect(sphere, ray, &dis, &normal))
    {
        color = vec3(1) * max(0.0, -dot(normal, vec3(0,0,-1)));
    }

    surf2Dwrite(make_float4(color.x, color.y, color.z, 1), texRef, x*sizeof(float4), y);
}

#endif
