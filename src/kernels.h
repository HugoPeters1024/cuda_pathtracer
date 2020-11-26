#ifndef H_KERNELS
#define H_KERNELS

#include "constants.h"
#include "use_cuda.h"
#include "types.h"
#include "globals.h"



#ifdef __CUDA_ARCH__
template <typename T>
__device__ inline void swapc(T& left, T& right)
{
    T tmp = left;
    left = right;
    right = tmp;
}
#else
#define swapc std::swap
#endif

HYBRID inline float lambert(const float3 &v1, const float3 &v2)
{
#ifdef __CUDA_ARCH__
    return max(dot(v1,v2),0.0f);
#else
    return std::max(dot(v1,v2),0.0f);
#endif
}

HYBRID inline bool firstIsFar(const uint& split_plane, const Ray& ray)
{
    // I know it's hacky but damn is it fast
    return ((float*)&ray.direction)[split_plane] < 0;
}

HYBRID inline Material getColliderMaterial(const HitInfo& hitInfo)
{
    switch (hitInfo.primitive_type)
    {
        case TRIANGLE: return _GMaterials[_GTriangleData[hitInfo.primitive_id].material];
        case SPHERE:   return _GMaterials[_GSpheres[hitInfo.primitive_id].material];
        case PLANE:    return _GMaterials[_GPlanes[hitInfo.primitive_id].material];
    }
    assert(false);
}

HYBRID inline float3 getColliderNormal(const HitInfo& hitInfo, const float3& intersectionPoint)
{
    switch (hitInfo.primitive_type)
    {
        case TRIANGLE: {
            float3 normal = _GTriangleData[hitInfo.primitive_id].n0;
            return normal;
        }
        case SPHERE: {
            const Sphere& sphere = _GSpheres[hitInfo.primitive_id];
            return normalize(intersectionPoint - sphere.pos);
        }
        case PLANE: {
            return _GPlanes[hitInfo.primitive_id].normal;
        }
    }
    assert(false);
    return make_float3(0);
}

HYBRID bool raySphereIntersect(const Ray& ray, const Sphere& sphere, float& t)
{
    float3 OC = ray.origin - sphere.pos;
    float a = dot(ray.direction, ray.direction);
    float b = 2 * dot(ray.direction, OC);
    float c = dot(OC, OC) - sphere.radius * sphere.radius;
    float det = b*b - 4 * a *c;
    if (det < 0) return false;
    det = sqrt(det);
    float tmin = (-b - det) / (2*a);
    float tmax = (-b + det) / (2*a);
    t = tmin;
    if (tmin < 0) t = tmax;
    return tmax > 0;
}

HYBRID bool rayPlaneIntersect(const Ray& ray, const Plane& plane, float& t)
{
    float q = dot(normalize(ray.direction), plane.normal);
    if (abs(q)<EPS) return false;
    t = -(dot(ray.origin, plane.normal) + plane.d)/q;
    return t>0;
}

HYBRID bool rayBoxIntersect(const Ray& r, const Box& box, float& mint, float& maxt)
{
    int signs[3];
    float3 invdir = 1.0 / r.direction;
    signs[0] = (int)(invdir.x < 0);
    signs[1] = (int)(invdir.y < 0);
    signs[2] = (int)(invdir.z < 0);
    float3 bounds[2] { box.vmin, box.vmax };
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    bool ret = true;

    tmin = (bounds[signs[0]].x - r.origin.x) * invdir.x;
    tmax = (bounds[1-signs[0]].x - r.origin.x) * invdir.x;
    tymin = (bounds[signs[1]].y - r.origin.y) * invdir.y;
    tymax = (bounds[1-signs[1]].y - r.origin.y) * invdir.y;

    ret &= !((tmin > tymax) || (tymin > tmax));
    tmin = max(tymin, tmin);
    tmax = min(tymax, tmax);

    tzmin = (bounds[signs[2]].z - r.origin.z) * invdir.z;
    tzmax = (bounds[1-signs[2]].z - r.origin.z) * invdir.z;

    ret &= !((tmin > tzmax) || (tzmin > tmax));
    tmin = max(tzmin, tmin);
    tmax = min(tzmax, tmax);

    mint = tmin;
    maxt = tmax;
    return ret && tmax > 0;
}

HYBRID bool rayTriangleIntersect(const Ray& ray, const TriangleV& triangle, float& t)
{
    float3 v0v1 = triangle.v1 - triangle.v0;
    float3 v0v2 = triangle.v2 - triangle.v0;
    float3 pvec = cross(ray.direction, v0v2);
    float det = dot(v0v1, pvec);
    if (fabs(det) < 0.0001f) return false;
    float invDet = 1 / det;

    float3 tvec = ray.origin - triangle.v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray.direction, qvec) * invDet;
    if(v < 0 || u + v > 1) return false;

    t = dot(v0v2, qvec) * invDet;
    return t > 0;
}

// Test if a given bvh node intersects with the ray. This function does not update the
// hit info distance because intersecting the boundingBox does not guarantee intersection
// any meshes. Therefore, the processLeaf function will keep track of the distance. BoxTest
// does use HitInfo for an early exit.
HYBRID inline bool boxtest(const Box& box, const Ray& ray, const HitInfo& hitInfo)
{
    float tmin, tmax;
    // Constrain that the closest point of the box must be closer than a known intersection.
    // Otherwise not triangle inside this box or it's children will ever change the intersection point
    // and can thus be discarded
    return rayBoxIntersect(ray, box, tmin, tmax) && tmin < hitInfo.t;
}

/*
__device__ bool traverseBVHShadows(const Ray& ray)
{
    const uint STACK_SIZE = 30;
    __shared__ uint stack[STACK_SIZE];

    HitInfo hitInfo;
    hitInfo.intersected = false;
    hitInfo.t = ray.length;

    stack[0] = 0;
    uint size = 1;

    while(size > 0)
    {
        uint current_id = stack[size-1];
        size -= 1;

        BVHNode current = GBVH[current_id];
        bool test = boxtest(current.boundingBox, ray, &hitInfo);
        if (__all_sync(__activemask(), !test)) continue;

        if (current.isLeaf())
        {
            uint start = current.t_start;
            uint end = start + current.t_count();
            float t;
            for(uint i=start; i<end; i++)
            {
                if (rayTriangleIntersect(ray, GTriangles[i], &t, hitInfo.t) && t < hitInfo.t)
                {
                    return true;
                }
            }
        }
        else
        {

            size += 2;

            // push on the stack, first the far child
            uint near = current.child1;
            uint far = near + 1;
            if (firstIsFar(current, ray)) swapc(near, far);

            stack[size - 2]   = far;
            stack[size - 1]   = near;
        }

    }

    return false;
}
*/

HYBRID HitInfo traverseBVHStack(const Ray& ray, bool anyIntersection)
{
    HitInfo hitInfo;
    hitInfo.intersected = false;
    hitInfo.t = ray.length;

    for(int i=0; i<_GSpheres.size; i++)
    {
        float t;
        if (raySphereIntersect(ray, _GSpheres[i], t) && t < hitInfo.t)
        {
            hitInfo.intersected = true;
            hitInfo.primitive_id = i;
            hitInfo.primitive_type = SPHERE;
            hitInfo.t = t;
            if (anyIntersection) return hitInfo;
        }
    }

    for(int i=0; i<_GPlanes.size; i++)
    {
        float t;
        if (rayPlaneIntersect(ray, _GPlanes[i], t) && t < hitInfo.t)
        {
            hitInfo.intersected = true;
            hitInfo.primitive_id = i;
            hitInfo.primitive_type = PLANE;
            hitInfo.t = t;
            if (anyIntersection) return hitInfo;
        }
    }

    float light_t;
    // Any intersection is being used by shadow rays, but they can't intersect the light source
    if (!anyIntersection && raySphereIntersect(ray, DLight, light_t) && light_t < hitInfo.t)
    {
        hitInfo.t = light_t;
        hitInfo.intersected = true;
        hitInfo.primitive_type = LIGHT;
    }

    const uint STACK_SIZE = 18;
    uint stack[STACK_SIZE];
    uint size = 0;

    const BVHNode root = _GBVH[0];
    if (boxtest(root.boundingBox, ray, hitInfo)) stack[size++] = 0;

    while(size > 0)
    {
        uint current_id = stack[--size];
        const BVHNode current = _GBVH[current_id];

        if (current.isLeaf())
        {
            uint start = current.t_start;
            uint end = start + current.t_count();
            float t;
            for(uint i=start; i<end; i++)
            {
                if (rayTriangleIntersect(ray, _GTriangles[i], t) && t < hitInfo.t)
                {
                    hitInfo.intersected = true;
                    hitInfo.primitive_id = i;
                    hitInfo.primitive_type = TRIANGLE;
                    hitInfo.t = t;
                    if (anyIntersection) return hitInfo;
                }
            }
        }
        else
        {
            uint near_id = current.child1;
            uint far_id = current.child1 + 1;
            BVHNode near = _GBVH[near_id];
            BVHNode far = _GBVH[far_id];
            if (firstIsFar(current.split_plane(), ray)) {
                swapc(near_id, far_id);
                swapc(near, far);
            }

            // push on the stack, first the far child
            if (boxtest(far.boundingBox, ray, hitInfo)) stack[size++] = far_id;
            if (boxtest(near.boundingBox, ray, hitInfo)) stack[size++] = near_id;

            //assert (size < STACK_SIZE);
        }
    }


    return hitInfo;
}

__device__ float3 BRDF(const float3& normal, uint& seed)
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

__device__ float3 BRDFUniform(const float3& normal, uint& seed)
{
    float3 r = normalize(make_float3(rand(seed) * 2 - 1, rand(seed) * 2 -1, rand(seed) * 2 - 1));
    return dot(r, normal) < 0 ? -r : r;
}

HYBRID Ray getReflectRay(const Ray& ray, const float3& normal, const float3& intersectionPos)
{
    float3 newDir = reflect(ray.direction, -normal);
    return Ray(intersectionPos + EPS * newDir, newDir, ray.pixeli);
}

HYBRID Ray getRefractRay(const Ray& ray, const float3& normal, const float3& intersectionPos, const Material& material, bool inside, float& reflected)
{
    // calculate the eta based on whether we are inside
    float n1 = 1.0;
    float n2 = material.refractive_index;
    if (inside) swapc(n1, n2);
    float eta = n1 / n2;

    float costi = dot(normal, -ray.direction);
    float k = 1 - (eta* eta) * (1 - costi * costi);
    // Total internal reflection
    if (k < 0) {
        reflected = 1;
        return Ray(make_float3(0), make_float3(0), 0);
//        return getReflectRay(ray, normal, intersectionPos, material, seed);
    }

    float3 refractDir = normalize(eta * ray.direction + normal * (eta * costi - sqrt(k)));

    // fresnell equation for reflection contribution
    float sinti = sqrt(max(0.0f, 1.0f - costi - costi));
    float costt = sqrt(1 - eta * eta * sinti * sinti);
    float spol = (n1 * costi - n2 * costt) / (n1 * costi + n2 * costt);
    float ppol = (n1 * costt - n2 * costi) / (n1 * costt + n2 * costi);

    reflected = 0.5 * (spol * spol + ppol * ppol);
    //if (rand(seed) < reflected) return getReflectRay(ray, normal, intersectionPos, material, seed);

    return Ray(intersectionPos + EPS * refractDir, refractDir, ray.pixeli);
}

__device__ Ray getDiffuseRay(const Ray& ray, const float3 normal, const float3 intersectionPos, uint seed)
{
    float3 newDir = BRDF(normal, seed);
    return Ray(intersectionPos + EPS * newDir, newDir, ray.pixeli);
}


__global__ void kernel_clear_state(TraceState* state)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NR_PIXELS) return;
    state[i] = { make_float3(1.0f), make_float3(0.0f), make_float3(0), 0, false };
}

__global__ void kernel_generate_primary_rays(Camera camera, float time)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);
    uint seed = getSeed(x,y,time);
    Ray ray = camera.getRay(x,y,seed);
    DRayQueue.push(ray);
}

__global__ void kernel_extend(HitInfo* intersections, uint bounce)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= DRayQueue.size) return;
    const Ray ray = DRayQueue[i];
    HitInfo hitInfo = traverseBVHStack(ray, false);
    intersections[i] = hitInfo;
}


__global__ void kernel_shade(const HitInfo* intersections, TraceState* stateBuf, float time, int bounce)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DRayQueue.size) return;
    const Ray ray = DRayQueue[i];
    TraceState state = stateBuf[ray.pixeli];


    uint seed = getSeed(i,wang_hash(i),time);
    const HitInfo hitInfo = intersections[i];
    if (!hitInfo.intersected) {
        // We consider the skydome a lightsource in the set of random bounces
        state.accucolor += state.mask * make_float3(0.2, 0.3, 0.6);
        stateBuf[ray.pixeli] = state;
        return;
    }

    if (hitInfo.primitive_type == LIGHT) {
        if (_NEE)
        {
            if (state.fromSpecular || bounce == 0)
            {
                // the light can only be seen by primary bounces.
                // the rest happens through NEE. If we do encounter a light
                // in this ray we simply add nothing to the contribution and
                // terminate.
                state.accucolor += state.mask * DLight_Color;
                stateBuf[ray.pixeli] = state;
            }
        }
        else
        {
            state.accucolor += state.mask * DLight_Color;
            stateBuf[ray.pixeli] = state;
        }
        return;
    }


    float3 intersectionPos = ray.origin + hitInfo.t * ray.direction;
    const Material material = getColliderMaterial(hitInfo);
    float3 originalNormal = getColliderNormal(hitInfo, intersectionPos);
    bool inside = dot(ray.direction, originalNormal) > 0;
    float3 colliderNormal = inside ? -originalNormal : originalNormal;

    state.mask = state.mask * material.color;
    // we can terminate this path
    if (dot(state.mask, state.mask) < 0.01) return;
    state.currentNormal = colliderNormal;

    // Create a secondary ray either diffuse or reflected
    Ray secondary;
    float random = rand(seed);
    if (random < material.transmit)
    {
        state.fromSpecular = true;
        if (inside)
        {
            // Take away any absorpted light using Beer's law.
            // when leaving the object
            float3 c = material.absorption;
            state.mask = state.mask * make_float3(exp(-c.x * hitInfo.t), exp(-c.y *hitInfo.t), exp(-c.z * hitInfo.t));
        }

        // Ray can turn into a reflection ray due to fresnell
        float reflected;
        secondary = getRefractRay(ray, colliderNormal, intersectionPos, material, inside, reflected);
        if (rand(seed) < reflected) {
            secondary = getReflectRay(ray, colliderNormal, intersectionPos);
        }
        float3 noiseDir = BRDF(secondary.direction, seed);
        secondary.direction = secondary.direction * (1 - material.glossy) + material.glossy * noiseDir;

        // Make sure we do not cast shadow rays
        state.correction = 0;
    }
    else if (random - material.transmit < material.reflect)
    {
        state.fromSpecular = true;
        secondary = getReflectRay(ray, colliderNormal, intersectionPos);
        float3 noiseDir = BRDF(secondary.direction, seed);
        secondary.direction = secondary.direction * (1-material.glossy) + material.glossy * noiseDir;

        // Make sure we do not cast shadow rays
        state.correction = 0;
    }
    else 
    {
        state.fromSpecular = false;
        secondary = getDiffuseRay(ray, colliderNormal, intersectionPos, seed);

        // incoming direction of the light.
        float lambert = dot(secondary.direction, colliderNormal);
        state.mask = state.mask * 2.0f * lambert;

        if (_NEE)
        {
            // Correct for the lambert term, which does not affect the direct light
            // at this point radiance at this position for the direct light sample.
            state.correction = 1.0f / (2.0f * lambert);
        }
    }

    DRayQueueNew.push(secondary);
    stateBuf[ray.pixeli] = state;

    // Create a shadow ray if it isn't corrected away (by being a mirror for example)
    if (_NEE && !state.fromSpecular && state.correction > EPS)
    {
        float3 fromLight = normalize(intersectionPos - DLight.pos);
        // Sample the hemisphere from that point.
        float3 r = BRDFUniform(fromLight, seed);

        // From the center of the light, go to sample point
        // (by definition of the BRDF on the visible by the origin (if not occluded)
        float3 samplePoint = DLight.pos + DLight.radius * r;

        float3 shadowDir = intersectionPos - samplePoint;
        float shadowLength = length(shadowDir);
        shadowDir /= shadowLength;

        // otherwise we are our own occluder
        if (dot(colliderNormal, shadowDir) < 0)
        {
            // We invert the shadowrays to get coherent origins.
            Ray shadowRay(samplePoint + EPS * shadowDir, shadowDir, ray.pixeli);
            shadowRay.length = shadowLength - 2 * EPS;
            DShadowRayQueue.push(shadowRay);
        }
    }
}

// Traces the shadow rays
__global__ void kernel_connect(TraceState* stateBuf)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DShadowRayQueue.size) return;

    const Ray& shadowRay = DShadowRayQueue[i];
    TraceState state = stateBuf[shadowRay.pixeli];
    HitInfo hitInfo = traverseBVHStack(shadowRay, true);
    if (hitInfo.intersected) return;

    float r2 = shadowRay.length * shadowRay.length;
    float3 lightNormal = normalize(shadowRay.origin - DLight.pos);
    float cost1 = dot(lightNormal, shadowRay.direction);
    float cost2 = dot(state.currentNormal, -shadowRay.direction);
    float SA = (3.1415926 * DLight.radius * DLight.radius * cost1 * cost2) / r2;
    //float NL = lambert(-shadowRay.direction, state.currentNormal);

    state.accucolor += state.mask * state.correction * DLight_Color * SA;
    stateBuf[shadowRay.pixeli] = state;
}

__global__ void kernel_add_to_screen(const TraceState* stateBuf, cudaSurfaceObject_t texRef)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NR_PIXELS) return;
    const uint x = i % WINDOW_WIDTH;
    const uint y = i / WINDOW_WIDTH;

    float3 color = stateBuf[i].accucolor;
    float4 old_color_all;
    surf2Dread(&old_color_all, texRef, x*sizeof(float4), y);
    float3 old_color = make_float3(old_color_all.x, old_color_all.y, old_color_all.z);
    surf2Dwrite(make_float4(old_color + color, old_color_all.w+1), texRef, x*sizeof(float4), y);
}

__global__ void kernel_clear_screen(cudaSurfaceObject_t texRef)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);
    surf2Dwrite(make_float4(0), texRef, x*sizeof(float4), y);
}

#endif
