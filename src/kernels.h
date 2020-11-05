#ifndef H_KERNELS
#define H_KERNELS

#include "use_cuda.h"
#include "types.h"
#include "globals.h"

__device__ inline float lambert(const float3 &v1, const float3 &v2)
{
    return max(dot(v1,v2),0.0f);
}

__device__ bool firstIsNear(uint node_id, const Ray& ray)
{
    BVHNode node = GBVH[node_id];
    switch(node.split_plane)
    {
        case 0: return ray.direction.x > 0;
        case 1: return ray.direction.y > 0;
        case 2: return ray.direction.z > 0;
    }
    return false;
}

__device__ inline uint parent(uint node_id)
{
    return GBVH[node_id].parent;
}

__device__ inline uint nearChild(uint node_id, const Ray& ray)
{
    BVHNode node = GBVH[node_id];
    return firstIsNear(node_id, ray) ? node.child1 : node.child2;
}

__device__ inline uint farChild(uint node_id, const Ray& ray)
{
    BVHNode node = GBVH[node_id];
    return firstIsNear(node_id, ray) ? node.child2 : node.child1;
}

__device__ inline uint sibling(uint node_id)
{
    BVHNode parentNode = GBVH[parent(node_id)];
    return parentNode.child1 == node_id ? parentNode.child2 : parentNode.child1;
}


__device__ inline bool isLeaf(uint node_id)
{
    BVHNode node = GBVH[node_id];
    return node.t_count > 0;
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

__device__ bool rayBoxIntersect(const Ray& r, const Box& box, float* mint, float* maxt)
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

    *mint = tmin;
    *maxt = tmax;
    return tmax > 0;
}

__device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle, float* t)
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
    *t = (dot(-N,ray.origin) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (*t < 0) return false; // the triangle is behind

    // compute the intersection point using equation 1
    float3 P = ray.origin + *t * ray.direction;

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

// Test if a given bvh node intersects with the ray. This function does not update the
// hit info distance because intersecting the boundingBox does not guarantee intersection
// any meshes. Therefore, the processLeaf function will keep track of the distance. BoxTest
// does use HitInfo for an early exit.
__device__ inline bool boxtest(uint node_id, const Ray& ray, const HitInfo* hitInfo)
{
    BVHNode node = GBVH[node_id];
    float tmin, tmax;
    // Constrain that the closest point of the box must be closer than a known intersection
    return rayBoxIntersect(ray, node.boundingBox, &tmin, &tmax) && tmin < hitInfo->t;
}


// Performs intersection against a leaf node's triangles
__device__ void processLeaf(uint node_id, const Ray& ray, HitInfo* hitInfo)
{
    BVHNode node = GBVH[node_id];
    uint start = node.t_start;
    uint end = start + node.t_count;
    for(uint i=start; i<end; i++)
    {
        float t;
        if (rayTriangleIntersect(ray ,GTriangles[i], &t) && t < hitInfo->t)
        {
            hitInfo->intersected = true;
            hitInfo->triangle_id = i;
            hitInfo->t = t;
            hitInfo->normal = GTriangles[i].n0;
        }
    }
    if (dot(hitInfo->normal, ray.direction) >= 0) hitInfo->normal = -hitInfo->normal;
}

// Stackless BVH traversal states
#define FROM_PARENT 0
#define FROM_SIBLING 1
#define FROM_CHILD 2


__device__ HitInfo traverseBVH(const Ray& ray)
{
    HitInfo hitInfo;
    hitInfo.intersected = false;
    hitInfo.t = 999999;

    uint root = 0;
    uint current = nearChild(root, ray);
    char state = FROM_PARENT;

    // BUG 400 iterations should be enough for the teapot
    // Handy fact: process leaf accounts for almost no computation.
    int zz = 0;
    while(zz++ < 400) {
        switch(state)
        {
            case FROM_CHILD:
                if (current == 0) return hitInfo; // finished;
                if (current == nearChild(parent(current), ray)) {
                    current=sibling(current); state=FROM_SIBLING;
                } else {
                    current = parent(current); state=FROM_CHILD;
                }
                break;

            case FROM_SIBLING:
            case FROM_PARENT:
                if (!boxtest(current,ray, &hitInfo)){
                    current = state == FROM_SIBLING ? parent(current) : sibling(current);
                    state = state == FROM_SIBLING ? FROM_CHILD : FROM_SIBLING;
                } else if (isLeaf(current)) {
                    processLeaf(current, ray, &hitInfo);
                    current = state == FROM_SIBLING ? parent(current) : sibling(current);
                    state = state == FROM_SIBLING ? FROM_CHILD : FROM_SIBLING;
                } else {
                    current = nearChild(current, ray); state=FROM_PARENT;
                }
                break;
        }
    }
    assert (zz < 400);
    return hitInfo;
}

__device__ float3 radiance(const Ray& ray)
{
    HitInfo hitInfo = traverseBVH(ray);
    if (hitInfo.intersected)
    {
        return make_float3(1);
    }
    return make_float3(0);
}


__global__ void kernel_pathtracer(cudaSurfaceObject_t texRef, float time, Camera camera) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    Ray ray = camera.getRay(x,y);
    float3 color = radiance(ray);
    surf2Dwrite(make_float4(color,1), texRef, x*sizeof(float4), y);
}

#endif
