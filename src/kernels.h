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
    return firstIsNear(node_id, ray) ? node_id + 1 : GBVH[node_id].child2;
}

__device__ inline uint farChild(uint node_id, const Ray& ray)
{
    return firstIsNear(node_id, ray) ? GBVH[node_id].child2 : node_id + 1;
}

__device__ inline uint sibling(uint node_id)
{
    uint parent_id = parent(node_id);
    return parent_id + 1 == node_id ? GBVH[parent_id].child2 : parent_id + 1;
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
    bool ret = true;
 
    tmin = (bounds[r.signs[0]].x - r.origin.x) * r.invdir.x; 
    tmax = (bounds[1-r.signs[0]].x - r.origin.x) * r.invdir.x; 
    tymin = (bounds[r.signs[1]].y - r.origin.y) * r.invdir.y; 
    tymax = (bounds[1-r.signs[1]].y - r.origin.y) * r.invdir.y; 
 
    ret &= !((tmin > tymax) || (tymin > tmax));
    tmin = max(tymin, tmin);
    tmax = min(tymax, tmax);
 
    tzmin = (bounds[r.signs[2]].z - r.origin.z) * r.invdir.z; 
    tzmax = (bounds[1-r.signs[2]].z - r.origin.z) * r.invdir.z; 
 
    ret &= !((tmin > tzmax) || (tzmin > tmax));
    tmin = max(tzmin, tmin);
    tmax = min(tzmax, tmax);

    *mint = tmin;
    *maxt = tmax;
    return ret && tmax > 0;
}

__device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle, float* t)
{
    bool ret = true;
    // compute plane's normal
    float3 v0v1 = triangle.v1 - triangle.v0;
    float3 v0v2 = triangle.v2 - triangle.v0;
    // no need to normalize
    float3 N = cross(v0v1,v0v2); // N
    float denom = dot(N,N);

    // Step 1: finding P

    // check if ray and plane are parallel ?
    float NdotRayDirection = dot(N,ray.direction);
    if (abs(NdotRayDirection) < 0.001f) // almost 0
        return false; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    float d = dot(N,triangle.v0);

    // compute t (equation 3)
    *t = (dot(-N,ray.origin) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    ret &= *t >= 0; // the triangle is behind

    // compute the intersection point using equation 1
    float3 P = ray.origin + *t * ray.direction;

    // Step 2: inside-outside test
    float3 C; // vector perpendicular to triangle's plane

    // edge 0
    float3 edge0 = triangle.v1 - triangle.v0;
    float3 vp0 = P - triangle.v0;
    C = cross(edge0, vp0);
    ret &= dot(N, C) >= 0; // P is on the right side

    // edge 1
    float3 edge1 = triangle.v2 - triangle.v1;
    float3 vp1 = P - triangle.v1;
    C = cross(edge1, vp1);
    ret &= dot(N,C) >= 0; // P is on the right side

    // edge 2
    float3 edge2 = triangle.v0 - triangle.v2;
    float3 vp2 = P - triangle.v2;
    C = cross(edge2, vp2);
    ret &= dot(N, C) >= 0; // P is on the right side;

    // we hit the triangle.
    return ret;
}

// Test if a given bvh node intersects with the ray. This function does not update the
// hit info distance because intersecting the boundingBox does not guarantee intersection
// any meshes. Therefore, the processLeaf function will keep track of the distance. BoxTest
// does use HitInfo for an early exit.
__device__ inline bool boxtest(uint node_id, const Ray& ray, const HitInfo* hitInfo)
{
    BVHNode node = GBVH[node_id];
    float tmin, tmax;
    // Constrain that the closest point of the box must be closer than a known intersection.
    // Otherwise not triangle inside this box or it's children will ever change the intersection point
    // and can thus be discarded
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
        if (rayTriangleIntersect(ray, GTriangles[i], &t) && t < hitInfo->t)
        {
            hitInfo->intersected = true;
            hitInfo->triangle_id = i;
            hitInfo->t = t;
        }
    }
}

__device__ HitInfo traverseBVHStack(const Ray& ray)
{
    HitInfo hitInfo;
    hitInfo.intersected = false;
    hitInfo.t = 999999;

    const uint STACK_SIZE = 200;
    uint stack[STACK_SIZE];
    uint* stackPtr = stack;
    *stackPtr++ = 0;

    uint size = 1;
    do
    {
        uint current = *--stackPtr;
        size -= 1;
        if (boxtest(current, ray, &hitInfo))
        {
            if (isLeaf(current))
            {
                processLeaf(current, ray, &hitInfo);
            }
            else
            {
                uint near = nearChild(current, ray);
                uint far = sibling(near);
                // push on the stack, first the far child
                *stackPtr++ = far;
                *stackPtr++ = near;
                size += 2;
                assert (size < STACK_SIZE);
            }
        }
    } while (size > 0);

    if (hitInfo.intersected) {
        hitInfo.normal = GTriangles[hitInfo.triangle_id].n0;
        if (dot(hitInfo.normal, ray.direction) >= 0) hitInfo.normal = -hitInfo.normal;
    }
    return hitInfo;
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

    bool searching = true;
    int zz = 0;
    // set a limit so we don't crash the pc if something goes wrong.
    const int max_iterations = 900;
    while(zz++ < max_iterations && searching) {
        switch(state)
        {
            case FROM_CHILD:
                if (current == 0) searching=false; // finished;
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

    // If the loop terminates because of the max_iterations something went wrong, perhaps the
    // scene is too large and max_iterations needs to be increased
    // but be weary of crashing your system if you go too high.
    assert (zz < max_iterations);
    if (hitInfo.intersected) {
        hitInfo.normal = GTriangles[hitInfo.triangle_id].n0;
        if (dot(hitInfo.normal, ray.direction) >= 0) hitInfo.normal = -hitInfo.normal;
    }
    return hitInfo;
}

__device__ float3 radiance(const Ray& ray, Ray* shadowRay)
{
    HitInfo hitInfo = traverseBVHStack(ray);
    if (hitInfo.intersected)
    {
        Triangle& t = GTriangles[hitInfo.triangle_id];

        float3 intersectionPos = ray.origin + (hitInfo.t - EPS) * ray.direction;
        float3 lightPos = make_float3(3*sin(GTime),2,3*cos(GTime));
        float3 toLight = lightPos - intersectionPos;
        float lightDis = length(toLight);
        toLight = toLight / lightDis;
        float ambient = 0.2;

        float3 color = ambient * t.color;

        // We trace shadow rays in reverse to more coherent rays
        *shadowRay = makeRay(lightPos, -toLight);
        shadowRay->shadowTarget = intersectionPos;
        shadowRay->active=true;

        return color;
    }
    shadowRay->active = false;
    return make_float3(0);
}

__global__ void kernel_pathtracer(Ray* rays, cudaSurfaceObject_t texRef, float time, Camera camera) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    Ray ray = camera.getRay(x,y);
    float3 color = clamp(radiance(ray, &rays[x + y * WINDOW_WIDTH]), 0,1);
    surf2Dwrite(make_float4(color,1), texRef, x*sizeof(float4), y);
}

__global__ void kernel_shadows(Ray* rays, cudaSurfaceObject_t texRef) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    Ray ray = rays[x + y * WINDOW_WIDTH];
    if (!ray.active) return;
    HitInfo hitInfo = traverseBVHStack(ray);
    if (hitInfo.intersected)
    {
        float3 intersectionPos = ray.origin + (hitInfo.t - EPS) * ray.direction;
        float3 isectDelta = intersectionPos - ray.shadowTarget;
        // New intersection point is not our illumination target
        if (dot(isectDelta, isectDelta) > 0.001) return;
        float illumination = 25;
        float falloff = 1.0 / (hitInfo.t * hitInfo.t);
        float lam = lambert(-ray.direction, hitInfo.normal);

        float4 old_color;
        surf2Dread(&old_color, texRef, x*sizeof(float4), y);
        surf2Dwrite(old_color + make_float4(illumination) * falloff * lam, texRef, x*sizeof(float4), y);
    }

}

#endif
