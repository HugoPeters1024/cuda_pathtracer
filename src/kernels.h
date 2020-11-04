#ifndef H_KERNELS
#define H_KERNELS

#include "use_cuda.h"
#include "types.h"
#include "globals.h"



__device__ inline float lambert(const float3 &v1, const float3 &v2)
{
    return max(dot(v1,v2),0.0f);
}

__device__ Ray makeRay(float3 origin, float3 direction)
{
    Ray ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.invdir = 1.0 / ray.direction;
    ray.signs[0] = (int)(ray.invdir.x < 0);
    ray.signs[1] = (int)(ray.invdir.y < 0);
    ray.signs[2] = (int)(ray.invdir.z < 0);
    return ray;
}

__device__ bool firstIsNear(const BVH_Seq& node, const Ray& ray)
{
    switch(node.split_plane)
    {
        case 0: return ray.direction.x > 0;
        case 1: return ray.direction.y > 0;
        case 2: return ray.direction.z > 0;
    }
    return false;
}

__device__ inline uint nearChild(const BVH_Seq& node, const Ray& ray)
{
    return firstIsNear(node, ray) ? node.child1 : node.child2;
}

__device__ inline uint farChild(const BVH_Seq& node, const Ray& ray)
{
    return firstIsNear(node, ray) ? node.child2 : node.child1;
}

__device__ inline uint sibling(const BVH_Seq& node)
{
    //return node.parent.child1 == node_id ? parentNode.child2 : parentNode.child1;
    return 0;
}

__device__ Ray getRayForPixel(unsigned int x, unsigned int y)
{
    float xf = 2 * (x / (float)WINDOW_WIDTH) - 1;
    float yf = 2 * (y / (float)WINDOW_HEIGHT) - 1;
    float ar = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;

    float camDis = 1.0;
    float3 pixel = make_float3(xf * ar, camDis, yf);
    float3 eye = make_float3(0);
    return makeRay(eye, normalize(pixel - eye));
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

__device__ bool rayBoxIntersect(const Ray& r, const Box& box, float* dis)
{
    float3 bounds[2] { box.vmin, box.vmax };
    float tmin, tmax, tymin, tymax, tzmin, tzmax; 
 
    tmin = (bounds[r.signs[0]].x - r.origin.x) * r.invdir.x; 
    tmax = (bounds[1-r.signs[0]].x - r.origin.x) * r.invdir.x; 
    tymin = (bounds[r.signs[1]].y - r.origin.y) * r.invdir.y; 
    tymax = (bounds[1-r.signs[1]].y - r.origin.y) * r.invdir.y; 
 
    if ((tmin > tymax) || (tymin > tmax)) 
        return false; 
    if (tymin > tmin) 
        tmin = tymin; 
    if (tymax < tmax) 
        tmax = tymax; 
 
    tzmin = (bounds[r.signs[2]].z - r.origin.z) * r.invdir.z; 
    tzmax = (bounds[1-r.signs[2]].z - r.origin.z) * r.invdir.z; 
 
    if ((tmin > tzmax) || (tzmin > tmax)) 
        return false; 
    if (tzmin > tmin) 
        tmin = tzmin; 
    if (tzmax < tmax) 
        tmax = tzmax; 

    *dis = tmax;
    return tmax > 0;
}

__device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle)
{
        // compute plane's normal
    float3 v0v1 = triangle.v1 - triangle.v0;
    float3 v0v2 = triangle.v2 - triangle.v0;
    // no need to normalize
    float3 N = cross(v0v1,v0v2); // N
    float denom = dot(N,N);

    // Step 1: finding P

    // check if ray and plane are parallel ?
    float NdotRayDirection = dot(N,ray.direction);
    if (abs(NdotRayDirection) < 0.0001f) // almost 0
        return false; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    float d = dot(N,triangle.v0);

    // compute t (equation 3)
    float t = (dot(-N,ray.origin) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind

    // compute the intersection point using equation 1
    float3 P = ray.origin + t * ray.direction;

    // Step 2: inside-outside test
    float3 C; // vector perpendicular to triangle's plane

    // edge 0
    float3 edge0 = triangle.v1 - triangle.v0;
    float3 vp0 = P - triangle.v0;
    C = cross(edge0, vp0);
    if (dot(N, C) < 0) return false; // P is on the right side

    // edge 1
    float3 edge1 = triangle.v2 - triangle.v1;
    float3 vp1 = P - triangle.v1;
    C = cross(edge1, vp1);
    if (dot(N,C) < 0)  return false; // P is on the right side

    // edge 2
    float3 edge2 = triangle.v0 - triangle.v2;
    float3 vp2 = P - triangle.v2;
    C = cross(edge2, vp2);
    if (dot(N, C) < 0) return false; // P is on the right side;

    // we hit the triangle.
    return true;

}

__device__ bool raySceneIntersect(const Ray& ray, const Box* spheres, int n, HitInfo* isectData)
{
    float dis = 1000;
    bool ret = false;
    int sphere_id = -1;

    for(int i=0; i<n; i++)
    {
        float newt;
        if (rayBoxIntersect(ray, spheres[i], &newt) && newt < dis)
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
        *isectData = HitInfo {
            spheres + sphere_id,
            dis,
            normalize(pos - spheres[sphere_id].centroid()),
            pos
        };
    }
    return true;

}

__device__ float3 radiance(Ray ray, const Box* spheres, int n) 
{
    float3 lightPos = make_float3(0,0,3);
    float3 sphereColor = make_float3(1,0,1);

    HitInfo isectData;
    if (!raySceneIntersect(ray, spheres, n, &isectData)) return make_float3(0);

    // Cast a shadow ray
    float3 toLight = lightPos - isectData.pos;
    Ray shadowRay = makeRay(isectData.pos, normalize(toLight));
    float distToLight2 = dot(toLight, toLight);

    // ray doens't reach light source
    HitInfo shadowData;
    if (raySceneIntersect(shadowRay, spheres, n, &shadowData) && shadowData.t * shadowData.t < distToLight2) return make_float3(0);

    return sphereColor * lambert(isectData.normal, shadowRay.direction);
}


__global__ void kernel_pathtracer(cudaSurfaceObject_t texRef, Triangle* triangles, int n, float time) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    Ray ray = getRayForPixel(x,y);

    for(int i=n; i<n+200; i++)
    {
        BVH_Seq node = BVH_Data[i];
        for(int t=node.t_start; t<node.t_start+node.t_count; t++)
        {
            Triangle triangle = triangles[t];
            if (rayTriangleIntersect(ray, triangle))
            {
                surf2Dwrite(make_float4(1,1,1,1), texRef, x*sizeof(float4), y);
                return;
            }
        }

    }
}

#endif
