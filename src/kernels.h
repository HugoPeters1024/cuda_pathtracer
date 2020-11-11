#ifndef H_KERNELS
#define H_KERNELS

#include "use_cuda.h"
#include "types.h"
#include "globals.h"

__device__ uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ inline float rand(uint* seed)
{
    uint m = wang_hash(*seed);
    *seed = m;

    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = reinterpret_cast<float&>(m);       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

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

__device__ HitInfo traverseBVHStack(const Ray& ray, bool ignoreLight)
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

    float light_t;
    if (!ignoreLight && raySphereIntersect(ray, GLight, &light_t) && light_t < hitInfo.t)
    {
        hitInfo.t = light_t;
        float3 isectPos = ray.origin + light_t * ray.direction;
        hitInfo.normal = normalize(isectPos - GLight.pos);
        hitInfo.intersected = true;
        hitInfo.triangle_id = 0;
    }

    return hitInfo;
}

__device__ float3 BRDF(const float3& normal, uint* seed)
{
    float r1 = 2 * 3.1415926535 * rand(seed);
    float r2 = rand(seed);
    float r2s = sqrtf(r2);

    float3 w = normal;
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
    float3 v = cross(w,u);

    // compute random ray direction on hemisphere using polar coordinates
    // cosine weighted importance sampling (favours ray directions closer to normal direction)
    return normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));
}


__device__ float3 sampleLight(const float3& origin, uint* seed)
{
    // The normal pointing to our origin from the light
    float3 normal = normalize(origin - GLight.pos);

    // Sample the brdf from that point.
    float3 r = BRDF(normal, seed);

    // From the center of the light, go to sample point
    // (by definition of the BRDF on the visible by the origin (if not occluded)
    float3 samplePoint = GLight.pos + GLight.radius * r * 1.001;

    // We invert the ray direction to maintain a higher level of coherence.
    float3 fromSample = origin - samplePoint;
    float3 newDir = normalize(fromSample);
    Ray ray = makeRay(samplePoint, newDir);

    HitInfo hitInfo = traverseBVHStack(ray, true);

    if (!hitInfo.intersected) return make_float3(0);
    float3 intersectionPos = ray.origin + (hitInfo.t - EPS) * ray.direction;
    float3 isectDelta = intersectionPos - origin;
    // New intersection point is not our illumination target
    if (dot(isectDelta, isectDelta) > 0.1) return make_float3(0);

    // solid angle of a sphere: https://en.wikipedia.org/wiki/Solid_angle
    float r2 = dot(fromSample, fromSample);
    float SA = 4 * 3.1415926535 * r2;
    return make_float3(25) / SA;
}

__device__ float3 radiance(Ray& ray, uint max_bounces, uint* seed)
{
    float3 accucolor = make_float3(0);
    float3 mask = make_float3(1);

    for(int bounces=0; bounces < max_bounces; bounces++)
    {
        HitInfo hitInfo = traverseBVHStack(ray, false);
        if (!hitInfo.intersected) return make_float3(0);
        if (hitInfo.triangle_id == 0) return make_float3(1);

        Triangle& collider = GTriangles[hitInfo.triangle_id];

        // setup ray for new intersection
        float3 newOrigin = ray.origin + (hitInfo.t - 0.001) * ray.direction;
        float3 newDir = BRDF(hitInfo.normal, seed);
        ray = makeRay(newOrigin, newDir);

        mask *= collider.color;
        mask *= dot(newDir, hitInfo.normal);
        accucolor += mask * sampleLight(newOrigin, seed);
    }

    return accucolor;
}

__global__ void kernel_clear_screen(cudaSurfaceObject_t texRef)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);
    surf2Dwrite(make_float4(0), texRef, x*sizeof(float4), y);
}


__global__ void kernel_pathtracer(Ray* rays, cudaSurfaceObject_t texRef, float time, uint max_bounces, Camera camera) {
    // Let's take it easy here cowboy
    assert (max_bounces < 5);
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);

    uint seed = (x + WINDOW_WIDTH * y) * (uint)(time * 100);
    Ray ray = camera.getRay(x,y);

    float3 color = radiance(ray, max_bounces,  &seed);
    float4 old_color_all;
    surf2Dread(&old_color_all, texRef, x*sizeof(float4), y);
    float3 old_color = make_float3(old_color_all.x, old_color_all.y, old_color_all.z);
    surf2Dwrite(make_float4(old_color + color, old_color_all.w+1), texRef, x*sizeof(float4), y);
}

#endif
