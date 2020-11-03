#ifndef H_KERNELS
#define H_KERNELS

#include "use_cuda.h"
#include "types.h"


__global__ void kernel_create_rays(Ray* rays) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    Vector3f eye(0,0,0);

    float xf = x / (float)WINDOW_WIDTH;
    float yf = y / (float)WINDOW_HEIGHT;

    float camDis = 0.4;
    Vector3f pixel(2*xf-1, camDis, 2*yf-1);

    const unsigned int i = dim1(x,y);
    rays[i] = Ray {
        eye,
        normalized(pixel - eye),
    };
}

__global__ void kernel_pathtracer(cudaSurfaceObject_t texRef, Ray* rays, Sphere* spheres, float time) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);
    const unsigned int i = dim1(x,y);
    Ray ray = rays[i];
    Sphere sphere = spheres[0];
    sphere.pos.x += sin(time);


    Vector3f color(0,0,0);

    float dis;
    Vector3f normal;
    if (raySphereIntersect(sphere, ray, &dis, &normal))
    {
        color = Vector3f(1) * max(0.0, -dot(normal, Vector3f(0,0,-1)));
    }

    surf2Dwrite(make_float4(color.x, color.y, color.z, 1), texRef, x*sizeof(float4), y);
}

#endif
