#ifndef H_KERNELS
#define H_KERNELS

#include "use_cuda.h"
#include "types.h"

__device__ inline float lambert(const float3 &v1, const float3 &v2)
{
    return max(dot(v1,v2),0.0f);
}

__device__ Ray getRayForPixel(unsigned int x, unsigned int y)
{
    float xf = 2 * (x / (float)WINDOW_WIDTH) - 1;
    float yf = 2 * (y / (float)WINDOW_HEIGHT) - 1;
    float ar = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;

    float camDis = 1.0;
    float3 pixel = make_float3(xf * ar, camDis, yf);
    float3 eye = make_float3(0);
    return Ray {
        eye,
        normalize(pixel - eye)
    };
}

__device__ bool raySphereIntersect(const Ray& ray, const Sphere& sphere, float* dis)
{
    float3 c = sphere.pos - ray.origin;
    float t = dot(c, ray.direction);
    float3 q = c - (t * ray.direction);
    float p2 = dot(q,q);
    if (p2 > sphere.radius * sphere.radius) return false;
    t -= sqrtf(sphere.radius * sphere.radius - p2);

    *dis = t;
    return t > 0;
}

__device__ bool raySceneIntersect(const Ray& ray, const Sphere* spheres, int n, Isect* isectData)
{
    float dis = 1000;
    bool ret = false;
    int sphere_id = -1;

    for(int i=0; i<n; i++)
    {
        float newt;
        if (raySphereIntersect(ray, spheres[i], &newt) && newt < dis)
        {
            dis = newt;
            ret = true;
            sphere_id = i;
        }

    }

    if (!ret) return false;

    float3 pos = ray.origin + (dis-0.001) * ray.direction;
    if (isectData != nullptr)
    {
        *isectData = Isect {
            spheres + sphere_id,
            dis,
            normalize(pos - spheres[sphere_id].pos),
            pos
        };
    }
    return true;

}

__device__ float3 radiance(Ray ray, const Sphere* spheres, int n) 
{
    float3 lightPos = make_float3(0,0,3);
    float3 sphereColor = make_float3(1,0,1);

    Isect isectData;
    if (!raySceneIntersect(ray, spheres, n, &isectData)) return make_float3(0);

    // Cast a shadow ray
    Ray shadowRay;
    float3 toLight = lightPos - isectData.pos;
    shadowRay.direction = normalize(toLight);
    shadowRay.origin = isectData.pos;
    float distToLight2 = dot(toLight, toLight);

    // ray doens't reach light source
    Isect shadowData;
    if (raySceneIntersect(shadowRay, spheres, n, &shadowData) && shadowData.t * shadowData.t < distToLight2) return make_float3(0);

    return sphereColor * lambert(isectData.normal, shadowRay.direction);
}


__global__ void kernel_pathtracer(cudaSurfaceObject_t texRef, Sphere* spheres, int n, float time) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    Ray ray = getRayForPixel(x,y);
    spheres[0].pos.x += 0.0001 * cos(time);

    float3 color = radiance(ray, spheres, n);
    surf2Dwrite(make_float4(color, 1), texRef, x*sizeof(float4), y);
}

#endif
