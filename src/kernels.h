#ifndef H_KERNELS
#define H_KERNELS

#include "use_cuda.h"
#include "types.h"
#include "globals.h"

__device__ inline float lambert(const float3 &v1, const float3 &v2)
{
    return max(dot(v1,v2),0.0f);
}

__device__ inline bool firstIsNear(const BVHNode& node, const Ray& ray)
{
    return (ray.direction.x > 0) * (node.split_plane == 0) + (ray.direction.y > 0) * (node.split_plane == 1) + (ray.direction.z > 0) * (node.split_plane == 2);
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
    if(dot(N, C) < 0) return false; // P is on the right side

    // edge 1
    float3 edge1 = triangle.v2 - triangle.v1;
    float3 vp1 = P - triangle.v1;
    C = cross(edge1, vp1);
    if(dot(N,C) < 0) return false; // P is on the right side

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
__device__ inline bool boxtest(const Box& box, const Ray& ray, const HitInfo* hitInfo)
{
    float tmin, tmax;
    // Constrain that the closest point of the box must be closer than a known intersection.
    // Otherwise not triangle inside this box or it's children will ever change the intersection point
    // and can thus be discarded
    return rayBoxIntersect(ray, box, &tmin, &tmax) && tmin < hitInfo->t;
}


// Performs intersection against a leaf node's triangles
__device__ void processLeaf(const BVHNode& node, const Ray& ray, HitInfo* hitInfo)
{
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
        uint current_id = *--stackPtr;
        BVHNode current = GBVH[current_id];
        size -= 1;
        if (boxtest(current.boundingBox, ray, &hitInfo))
        {
            if (current.isLeaf())
            {
                processLeaf(current, ray, &hitInfo);
            }
            else
            {
                bool firstNear = firstIsNear(current, ray);
                uint near = firstNear ? current_id + 1 : current.child2;
                uint far = firstNear ? current.child2 : current_id + 1;
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


__device__ float3 radiance(const Ray& ray, Ray* shadowRay, uint depth)
{
    if (depth > 2) return make_float3(0);
    HitInfo hitInfo = traverseBVHStack(ray);
    if (hitInfo.intersected)
    {
        Triangle& t = GTriangles[hitInfo.triangle_id];

        float3 intersectionPos = ray.origin + (hitInfo.t - EPS) * ray.direction;
        float3 lightPos = make_float3(3*sin(GTime),2,3*cos(GTime));
        float3 toLight = lightPos - intersectionPos;
        float lightDis = length(toLight);
        toLight = toLight / lightDis;

        // ambient comes from above
        float ambient = 0.2 * dot(hitInfo.normal, make_float3(0,1,0));

        float3 color = ambient * t.color;

        // We trace shadow rays in reverse to more coherent rays
        // but only if the light source isn't behind the triangle
        if (dot(toLight, hitInfo.normal) > 0) {
            *shadowRay = makeRay(lightPos, -toLight);
            shadowRay->shadowTarget = intersectionPos;
            shadowRay->active = true;
        }
        else shadowRay->active = false;

        return color;
    }
    shadowRay->active = false;
    return make_float3(0);
}

__device__ static float getRandom(uint *seed0, uint *seed1) {
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    // Convert to float
    union {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}

__device__ float3 radiance2(Ray& ray, uint* s0, uint* s1)
{
    float3 accucolor = make_float3(0);
    float3 mask = make_float3(1);

    for(int bounces=0; bounces < 2; bounces++)
    {
        HitInfo hitInfo = traverseBVHStack(ray);
        if (!hitInfo.intersected) return make_float3(0);

        Triangle& collider = GTriangles[hitInfo.triangle_id];
        float3 emission = (collider.color.y < 0.9) * make_float3(100);
        accucolor += mask * emission;

        float r1 = 2 * 3.1415926535 * getRandom(s0, s1);
        float r2 = getRandom(s0, s1);
        float r2s = sqrtf(r2);

        float3 w = hitInfo.normal;
        float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
        float3 v = cross(w,u);

        // compute random ray direction on hemisphere using polar coordinates
        // cosine weighted importance sampling (favours ray directions closer to normal direction)
        float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

        // setup ray for new intersection
        float3 newOrigin = ray.origin + (hitInfo.t - 0.001) * ray.direction;
        ray = makeRay(newOrigin, d);

        mask *= collider.color;
        mask *= dot(d, hitInfo.normal);
        mask *= 2;
    }

    return accucolor;
}

__global__ void kernel_pathtracer(Ray* rays, cudaSurfaceObject_t texRef, float time, Camera camera) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    uint s0 = x * int(time*100);
    uint s1 = y * int(time*100);

    Ray ray = camera.getRay(x,y);

    float alpha = 0.95;
    float3 color = clamp(radiance2(ray, &s0, &s1), 0,1);
    float4 old_color;
    surf2Dread(&old_color, texRef, x*sizeof(float4), y);
    surf2Dwrite(old_color * alpha + make_float4(color,1)*(1-alpha), texRef, x*sizeof(float4), y);
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
        float3 color = GTriangles[hitInfo.triangle_id].color;

        float4 old_color;
        surf2Dread(&old_color, texRef, x*sizeof(float4), y);
        surf2Dwrite(old_color + make_float4(color,1) * illumination * falloff * lam, texRef, x*sizeof(float4), y);
    }

}

#endif
