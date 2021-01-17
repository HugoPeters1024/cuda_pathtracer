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

HYBRID inline float rand(RandState& randState)
{
#ifdef __CUDA_ARCH__
    if (randState.sampleIdx < 100)
    {
        randState.blueNoiseOffset += make_float2(rand(randState.seed), rand(randState.seed));
        const float2 uv = randState.kernelPos + randState.blueNoiseOffset;
        return tex2D<float>(randState.blueNoise, uv.x, uv.y);
    }
#endif
    return rand(randState.seed);
}

HYBRID inline float2 normalToUv(const float3& n)
{
    float theta = atan2f(n.x, n.z) / (2.0f * PI);
    float phi = -acosf(n.y) / PI;
    return make_float2(theta, phi);
}

// Uv range: [0, 1]
HYBRID inline float3 uvToNormal(const float2& uv)
{
    float theta = uv.x * 2.0f * PI;
    float phi = -uv.y * PI;

    float3 n;
    n.z = cosf(theta) * sinf(phi);
    n.x = sinf(theta) * sinf(phi);
    n.y = cosf(phi);
    return n;
}


HYBRID inline uint binarySearch(const float* values, const uint& nrValues, const float& target)
{
    uint L = 0;
    uint R = nrValues - 1;
    uint m;
    while(L < R-1)
    {
        m = (L+R)/2;
        if (values[m] >= target)
            R = m;
        else if (values[m] < target)
            L = m;
    }
    return m;
}

HYBRID inline Ray transformRay(const Ray& ray, const glm::mat4x4& transform)
{
    glm::vec4 oldDir = glm::vec4(ray.direction.x, ray.direction.y, ray.direction.z, 0);
    glm::vec4 oldOrigin = glm::vec4(ray.origin.x, ray.origin.y, ray.origin.z, 1);

    glm::vec4 newDir = transform * oldDir;
    glm::vec4 newOrigin = transform * oldOrigin;

    Ray transformedRay = ray;
    transformedRay.direction = normalize(make_float3(newDir.x, newDir.y, newDir.z));
    transformedRay.origin = make_float3(newOrigin.x, newOrigin.y, newOrigin.z);
    return transformedRay;
}

HYBRID inline uint getColliderMaterialID(const HitInfo& hitInfo)
{
    switch (hitInfo.primitive_type)
    {
        case TRIANGLE: return _GVertexData[hitInfo.primitive_id].material;
        case SPHERE:   return _GSpheres[hitInfo.primitive_id].material;
        case PLANE:    return _GPlanes[hitInfo.primitive_id].material;
    }
    assert(false);
}

HYBRID inline float3 getColliderNormal(const HitInfo& hitInfo, const float3& intersectionPoint)
{
    switch (hitInfo.primitive_type)
    {
        case TRIANGLE: {
            return _GVertexData[hitInfo.primitive_id].normal;
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
    if (fabsf(a) < 0.001) return false;
    float b = 2 * dot(ray.direction, OC);
    float c = dot(OC, OC) - sphere.radius * sphere.radius;
    float det = b*b - 4 * a *c;
    if (det < 0) return false;
    det = sqrtf(det);
    float tmin = (-b - det) / (2*a);
    float tmax = (-b + det) / (2*a);
    t = tmin;
    if (tmin < 0) t = tmax;
    return tmax > 0;
}

HYBRID bool rayPlaneIntersect(const Ray& ray, const Plane& plane, float& t)
{
    float q = dot(normalize(ray.direction), plane.normal);
    if (fabsf(q)<EPS) return false;
    t = -(dot(ray.origin, plane.normal) + plane.d)/q;
    return t>0;
}

HYBRID inline bool slabTest(const float3& rayOrigin, const float3& invRayDir, const Box& box, float& tmin)
{
#ifdef __CUDA_ARCH__
    float3 t0 = (box.vmin - rayOrigin) * invRayDir;
    float3 t1 = (box.vmax - rayOrigin) * invRayDir;
    float3 tmin3 = fminf(t0,t1), tmax3 = fmaxf(t1,t0);
    tmin = fmaxcompf(tmin3);
    return fmincompf(tmax3) >= fmaxf(0.0f, tmin);
#else
    __m128 bmin = _mm_setr_ps(box.vmin.x, box.vmin.y, box.vmin.z, 0.0f);
    __m128 bmax = _mm_setr_ps(box.vmax.x, box.vmax.y, box.vmax.z, 0.0f);
    __m128 sdRayOrigin = _mm_setr_ps(rayOrigin.x, rayOrigin.y, rayOrigin.z, 0.0f);
    __m128 sdInvRayDir = _mm_setr_ps(invRayDir.x, invRayDir.y, invRayDir.z, 0.0f);
    __m128 t0 = _mm_mul_ps(_mm_sub_ps(bmin, sdRayOrigin), sdInvRayDir);
    __m128 t1 = _mm_mul_ps(_mm_sub_ps(bmax, sdRayOrigin), sdInvRayDir);
    __m128 sdtmin = _mm_min_ps(t0, t1);
    __m128 sdtmax = _mm_max_ps(t0, t1);
    tmin = fmaxf(fmaxf(sdtmin[0], sdtmin[1]), sdtmin[2]);
    float tmax = fminf(fminf(sdtmax[0], sdtmax[1]), sdtmax[2]);
    return tmax >= fmaxf(0.0f, tmin);
#endif
}

HYBRID inline bool rayTriangleIntersect(const Ray& ray, const TriangleV& triangle, float& t, float& u, float& v)
{
    float3 v0v1 = triangle.v1 - triangle.v0;
    float3 v0v2 = triangle.v2 - triangle.v0;
    float3 pvec = cross(ray.direction, v0v2);
    float det = dot(v0v1, pvec);
    if (fabsf(det) < 0.0001f) return false;
    float invDet = 1.0f / det;

    float3 tvec = ray.origin - triangle.v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, v0v1);
    v = dot(ray.direction, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    t = dot(v0v2, qvec) * invDet;
    return t > 0;
}

// Test if a given bvh node intersects with the ray. This function does not update the
// hit info distance because intersecting the boundingBox does not guarantee intersection
// any meshes. Therefore, the processLeaf function will keep track of the distance. BoxTest
// does use HitInfo for an early exit.
HYBRID inline bool boxtest(const Box& box, const float3& rayOrigin, const float3& invRayDir, float& tmin, const HitInfo& hitInfo)
{
    // Constrain that the closest point of the box must be closer than a known intersection.
    // Otherwise not triangle inside this box or it's children will ever change the intersection point
    // and can thus be discarded
    return slabTest(rayOrigin, invRayDir, box, tmin) && tmin < hitInfo.t;
}

template <bool anyIntersection>
HYBRID bool traverseBVHStack(const Ray& ray, HitInfo& hitInfo, const Instance& instance)
{
    float t, u, v;

    const uint STACK_SIZE = 18;
    uint stack[STACK_SIZE];
    uint* stackPtr = stack;

    // declare variables used in the loop
    float tnear, tfar;

    const float3 invRayDir = 1.0f / ray.direction;

    const Model& model = _GModels[instance.model_id];
    const TriangleV* vertices = _GVertices;
    BVHNode current = model.bvh[0];
    if (!boxtest(current.getBox(), ray.origin, invRayDir, tnear, hitInfo)) return false;

    bool needPop = false;
    bool intersected = false;

    while(true)
    {
        if (current.isLeaf())
        {
            const uint start = current.t_start();
            const uint end = start + current.t_count();
            for(uint i=start; i<end; i++)
            {
                if (rayTriangleIntersect(ray, vertices[i], t, u, v) && t < hitInfo.t)
                {
                    if (anyIntersection)
                    {
                        return true;
                    }
                    else
                    {
                        hitInfo.primitive_id = i;
                        hitInfo.t = t;
                        intersected = true;
                    }
                }
            }
            needPop = true;
        }
        else
        {
            const uint near_id = current.child1();
            const BVHNode& near = model.bvh[near_id];
            const BVHNode& far = model.bvh[near_id+1];
            const bool bnear = boxtest(near.getBox(), ray.origin, invRayDir, tnear, hitInfo);
            const bool bfar = boxtest(far.getBox(), ray.origin, invRayDir, tfar, hitInfo);

            if (bnear & bfar) {
                bool rev = tnear > tfar;
                current = rev ? far : near;
                *stackPtr++ = near_id + !rev;
            } else if (bnear) {
                current = near;
            } else if (bfar) {
                current = far;
            } else {
                needPop = true;
            }
            //assert (size < STACK_SIZE);
        }

        if (needPop) 
        {
            if (stack != stackPtr)
                current = model.bvh[*--stackPtr];
            else
                return intersected;
            needPop = false;
        }
    }
}

template <bool anyIntersection>
HYBRID HitInfo traverseTopLevel(const Ray& ray)
{
    HitInfo hitInfo;
    hitInfo.primitive_id = 0xffffffff;
    hitInfo.t = ray.length;

    float t, tmin;

    for(int i=0; i<_GSpheres.size; i++)
    {
        if (raySphereIntersect(ray, _GSpheres[i], t) && t < hitInfo.t)
        {
            if (anyIntersection)
            {
                hitInfo.primitive_id = i;
                return hitInfo;
            }
            else
            {
                hitInfo.primitive_id = i;
                hitInfo.primitive_type = SPHERE;
                hitInfo.t = t;
            }
        }
    }

    for(int i=0; i<_GPlanes.size; i++)
    {
        if (rayPlaneIntersect(ray, _GPlanes[i], t) && t < hitInfo.t)
        {
            if (anyIntersection)
            {
                hitInfo.primitive_id = i;
                return hitInfo;
            }
            else
            {
                hitInfo.primitive_id = i;
                hitInfo.primitive_type = PLANE;
                hitInfo.t = t;
            }
        }
    }

    float3 invRayDir = 1.0f / ray.direction;

    const uint STACK_SIZE = 5;
    uint stack[STACK_SIZE];
    uint size = 0;

    TopLevelBVH current;
    stack[size++] = 0;


    while(size > 0)
    {
        current = _GTopBVH[stack[--size]];
        if (!boxtest(current.box, ray.origin, invRayDir, tmin, hitInfo)) continue;

        if (current.isLeaf)
        {
            Instance instance = _GInstances[current.leaf];
            Ray transformedRay = transformRay(ray, instance.invTransform);
            if (traverseBVHStack<anyIntersection>(transformedRay, hitInfo, instance))
            {
                if (anyIntersection)
                {
                    // this will cause the intersection boolean to be true;
                    hitInfo.primitive_id = 0;
                    return hitInfo;
                }
                else
                {
                    hitInfo.primitive_type = TRIANGLE;
                    hitInfo.instance_id = current.leaf;
                }
            }
        }
        else
        {
            stack[size++] = current.child1;
            stack[size++] = current.child2;
        }
    }

    return hitInfo;
}


HYBRID float3 SampleHemisphereCosine(const float3& normal, const float& r0, const float& r1)
{
    float r = sqrtf(r0);
    float theta = 2 * PI * r1;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float3 sample =  make_float3( x, y, sqrtf(1 - r0));

    const float3& w = normal;
    float3 u = normalize(cross((fabsf(w.x) > .1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
    float3 v = normalize(cross(w,u));

    return normalize(make_float3(
            dot(sample, make_float3(u.x, v.x, w.x)),
            dot(sample, make_float3(u.y, v.y, w.y)),
            dot(sample, make_float3(u.z, v.z, w.z))));
}

HYBRID float3 SampleHemisphereCached(const float3& normal, RandState& randState, const TriangleD& triangleD, int& sampleBucket)
{
    const float sample = rand(randState);
    float acc = 0.0f;
    sampleBucket = -1;
    do
    {
        acc += triangleD.radianceCache[++sampleBucket];
    }
    while (acc < sample);

    const float r0Min = (sampleBucket < 4) ? 0 : 0.5;
    const float r0Max = (sampleBucket < 4) ? 0.5 : 1.0f;
    const uint r1i = sampleBucket % 4;
    const float r1Min = r1i * 0.25f;
    const float r1Max = (r1i+1.0f) * 0.25f;

    const float r0R = rand(randState);
    const float r1R = rand(randState);
    const float r0 = r0Min * r0R + r0Max * (1-r0R);
    const float r1 = r1Min * r1R + r1Max * (1-r1R);
    return SampleHemisphereCosine(normal, r0, r1);
}


HYBRID float3 SampleHemisphere(const float3& normal, RandState& randState)
{
    const float u1 = rand(randState);
    const float u2 = rand(randState);
    const float r = sqrtf(1.0f - u1 * u1);
    const float phi = 2.0f * PI * u2;
    const float3 sample = make_float3(cosf(phi)*r, sinf(phi)*r, u1);

    const float3& w = normal;
    float3 u = normalize(cross((fabsf(w.x) > .1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
    float3 v = normalize(cross(w,u));

    return normalize(make_float3(
            dot(sample, make_float3(u.x, v.x, w.x)),
            dot(sample, make_float3(u.y, v.y, w.y)),
            dot(sample, make_float3(u.z, v.z, w.z))));
}

HYBRID Ray getReflectRay(const Ray& ray, const float3& normal, const float3& intersectionPos)
{
    const float3 newDir = reflect(ray.direction, normal);
    return Ray(intersectionPos + EPS * newDir, newDir, ray.pixeli);
}

HYBRID Ray getRefractRay(const Ray& ray, const float3& normal, const float3& intersectionPos, const Material& material, const bool& inside, float& reflected)
{
    // calculate the eta based on whether we are inside
    const float n1 = inside ? material.refractive_index : 1.0f;
    const float n2 = inside ? 1.0f : material.refractive_index;
    const float eta = n1 / n2;

    const float costi = dot(normal, -ray.direction);
    const float k = 1 - (eta* eta) * (1 - costi * costi);
    // Total internal reflection
    if (k < 0) {
        reflected = 1;
        return Ray(make_float3(0), make_float3(0), 0);
    }

    const float3 refractDir = normalize(eta * ray.direction + normal * (eta * costi - sqrtf(k)));

    // fresnell equation for reflection contribution
    const float sinti = sqrtf(fmaxf(0.0f, 1.0f - costi - costi));
    const float costt = sqrtf(1.0f - eta * eta * sinti * sinti);
    const float spol = (n1 * costi - n2 * costt) / (n1 * costi + n2 * costt);
    const float ppol = (n1 * costt - n2 * costi) / (n1 * costt + n2 * costi);

    reflected = 0.5f * (spol * spol + ppol * ppol);
    return Ray(intersectionPos + EPS * refractDir, refractDir, ray.pixeli);
}

__global__ void kernel_clear_state(TraceStateSOA state)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NR_PIXELS) return;
    state.accucolors[i] = make_float4(0.0f);
    state.masks[i] = make_float4(1.0f,1.0f,1.0f,__int_as_float(1));
}

__global__ void kernel_generate_primary_rays(Camera camera, RandState randState)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);
    uint seed = getSeed(x,y,randState.randIdx);
    RayPacked ray = RayPacked(camera.getRay(x,y,seed));
    DRayQueue.push(ray);
}

__global__ void kernel_extend(HitInfoPacked* intersections, uint bounce)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= DRayQueue.size) return;
    const Ray ray = DRayQueue[i].getRay();
    HitInfo hitInfo = traverseTopLevel<false>(ray);
    intersections[i] = HitInfoPacked(hitInfo);
}


__global__ void kernel_shade(const HitInfoPacked* intersections, TraceStateSOA stateBuf, RandState randState, int bounce, CudaTexture skydome, CDF d_skydomeCDF, SampleCache* sampleCache)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DRayQueue.size) return;
    const Ray ray = DRayQueue[i].getRay();
    TraceState state = stateBuf.getState(ray.pixeli);

    const uint x = ray.pixeli % WINDOW_WIDTH;
    const uint y = ray.pixeli / WINDOW_WIDTH;
    const HitInfo hitInfo = intersections[i].getHitInfo();
    if (!hitInfo.intersected()) {
        // We consider the skydome a lightsource in the set of random bounces
        //if (!_NEE || state.fromSpecular) {
            float2 uvCoords = normalToUv(ray.direction);
            float3 sk = get3f(tex2D<float4>(skydome.texture_id, uvCoords.x, uvCoords.y));
            state.accucolor += state.mask * sk;
            stateBuf.setState(ray.pixeli, state);
       // }
        return;
    }

    randState.seed = getSeed(x,y,randState.randIdx);
    randState.kernelPos = make_float2((float)x, (float)y) / randState.blueNoiseSize;
    randState.blueNoiseOffset = make_float2(rand(randState.seed), rand(randState.seed));

    // Only triangles are always part of instances but that is the
    // responsibility of the code dereferencing the pointer.
    const Instance* instance = _GInstances + hitInfo.instance_id;
    const float3 intersectionPos = ray.origin + hitInfo.t * ray.direction;
    const uint material_id = getColliderMaterialID(hitInfo);
    Material material = _GMaterials[material_id];
    float3 surfaceNormal = getColliderNormal(hitInfo, intersectionPos);

    // invert the normal and position transformation back to world space
    if (hitInfo.primitive_type == TRIANGLE)
    {
        surfaceNormal = normalize(make_float3(instance->transform * glm::vec4(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z, 0.0f)));
    }

    bool inside = dot(ray.direction, surfaceNormal) > 0;
    surfaceNormal = inside ? -surfaceNormal : surfaceNormal;
    float3 colliderNormal = surfaceNormal;

    // Triangle is emmisive
    if (fmaxcompf(material.emission) > EPS)
    {
        if (!_NEE || state.fromSpecular)
        {
            state.accucolor += state.mask * material.emission;
            stateBuf.setState(ray.pixeli, state);
        }
        return;
    }

    if (hitInfo.primitive_type == PLANE) {
        uint px = (uint)(fabsf(intersectionPos.x/4 + 1000));
        uint py = (uint)(fabsf(intersectionPos.z/4 + 1000));
        material.diffuse_color = (px + py)%2 == 0 ? make_float3(1) : make_float3(0.2);
    }

    // sample the texture of the material by redoing the intersection
    if (material.hasTexture || material.hasNormalMap)
    {
        const TriangleD& triangleData = _GVertexData[hitInfo.primitive_id];
        const TriangleV& triangleV = _GVertices[hitInfo.primitive_id];

        float t, u, v;
        Ray transformedRay = transformRay(ray, instance->invTransform);
        assert( rayTriangleIntersect(transformedRay, triangleV, t, u, v));
        // Calculate the exact texture location by interpolating the three vertices' texture coords
        float2 uv = triangleData.uv0 * (1-u-v) + triangleData.uv1 * u + triangleData.uv2 * v;

        if (material.hasTexture)
        {
            // According to the mtl spec texture should be multiplied by the diffuse color
            // https://www.loc.gov/preservation/digital/formats/fdd/fdd000508.shtml
            material.diffuse_color = material.diffuse_color * get3f(tex2D<float4>(material.texture, uv.x, uv.y));
        }

        // sample the normal of the material by redoing the intersection
        if (material.hasNormalMap)
        {
            float4 texColor = tex2D<float4>(material.normal_texture, uv.x, uv.y);
            float3 texNormalT = normalize((get3f(texColor)*2)-make_float3(1));
            float3 texNormal = normalize(make_float3(
                    dot(texNormalT, make_float3(triangleData.tangent.x, triangleData.bitangent.x, triangleData.normal.x)),
                    dot(texNormalT, make_float3(triangleData.tangent.y, triangleData.bitangent.y, triangleData.normal.y)),
                    dot(texNormalT, make_float3(triangleData.tangent.z, triangleData.bitangent.z, triangleData.normal.z))
            ));

            // Transform the normal from model space to world space
            texNormal = normalize(make_float3(instance->transform * glm::vec4(texNormal.x, texNormal.y, texNormal.z, 0.0f)));
            colliderNormal = dot(texNormal, colliderNormal) < 0.0f ? -texNormal : texNormal;
        }
    }

    // Create a secondary ray either diffuse or reflected
    Ray secondary;
    float random = rand(randState);
    bool cullSecondary = false;
    const float3 BRDF = material.diffuse_color / PI;
    SampleCache cache;
    cache.sample_type = SAMPLE_IGNORE;

    if (random < material.transmit)
    {
        state.fromSpecular = true;
        if (inside)
        {
            // Take away any absorpted light using Beer's law.
            // when leaving the object
            const float3& c = material.absorption;
            state.mask *= make_float3(exp(-c.x * hitInfo.t), exp(-c.y *hitInfo.t), exp(-c.z * hitInfo.t));
        }

        // Ray can turn into a reflection ray due to fresnell
        float reflected;
        secondary = getRefractRay(ray, colliderNormal, intersectionPos, material, inside, reflected);
        if (rand(randState) < reflected) {
            state.mask *= material.diffuse_color;
            secondary = getReflectRay(ray, colliderNormal, intersectionPos);
        }
        float3 noiseDir = SampleHemisphereCosine(secondary.direction, rand(randState), rand(randState));
        secondary.direction = secondary.direction * (1.0f - material.glossy) + material.glossy * noiseDir;
    }
    else if (random - material.transmit < material.reflect)
    {
        state.fromSpecular = true;
        state.mask *= material.diffuse_color;
        secondary = getReflectRay(ray, colliderNormal, intersectionPos);
        float3 noiseDir = SampleHemisphereCosine(secondary.direction, rand(randState), rand(randState));
        secondary.direction = secondary.direction * (1.0f-material.glossy) + material.glossy * noiseDir;
    }
    else 
    {
        state.fromSpecular = false;

        // Create a shadow ray for diffuse objects
        if (_NEE)
        {
            // Skydome CDF
            /*
            float r = rand(seed);
            // binary search the cum values
            uint sample = binarySearch(d_skydomeCDF.cumValues, d_skydomeCDF.nrItems, r);
            uint x = sample % skydome.width;
            uint y = sample / skydome.width;
            float2 uv = make_float2((float)x / (float)skydome.width, (float)y / (float)skydome.height);
            float3 normal = uvToNormal(uv);
            if (dot(normal, colliderNormal) > 0)
            {
                float2 uv = normalToUv(normal);
                float3 lightSample = get3f(tex2D<float4>(skydome.texture_id, uv.x, uv.y));
                float PDF = 1.0f / (d_skydomeCDF.values[sample] * d_skydomeCDF.nrItems);
                state.light = state.mask * lightSample * BRDF * PI * PDF * 4.0f;

                float3 farOut = intersectionPos + 10000 * normal;
                Ray shadowRay(farOut, -normal, ray.pixeli);
                shadowRay.length = 10000 - EPS;
                DShadowRayQueue.push(shadowRay);
            }
            */

            // Choose an area light at random
            const uint lightSource = uint(fmin(rand(randState),0.99999999999f) * DTriangleLights.size);
            const TriangleLight& light = DTriangleLights[lightSource];
            const TriangleV& lightV = _GVertices[light.triangle_index];
            const TriangleD& lightD = _GVertexData[light.triangle_index];
            const glm::mat4x4& lightTransform = _GInstances[light.instance_index].transform;

            // transform the vertices to world space
            const float3 v0 = get3f(lightTransform * glm::vec4(lightV.v0.x, lightV.v0.y, lightV.v0.z, 1));
            const float3 v1 = get3f(lightTransform * glm::vec4(lightV.v1.x, lightV.v1.y, lightV.v1.z, 1));
            const float3 v2 = get3f(lightTransform * glm::vec4(lightV.v2.x, lightV.v2.y, lightV.v2.z, 1));

            const float3 v0v1 = v1 - v0;
            const float3 v0v2 = v2 - v0;
            const float3 cr = cross(v0v1, v0v2);
            const float crLength = length(cr);

            float u=rand(randState), v=rand(randState);
            if (u+v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
            const float3 samplePoint = v0 + u * v0v1 + v * v0v2;

            float3 shadowDir = intersectionPos - samplePoint;
            const float shadowLength = length(shadowDir);
            const float invShadowLength = 1.0f / shadowLength;
            shadowDir *= invShadowLength;

            const float3 lightNormal = cr / crLength;
            const float NL = dot(colliderNormal, -shadowDir);
            const float LNL = dot(lightNormal, shadowDir);

            // otherwise we are our own occluder or view the backface of the light
            if (NL > 0 && dot(-shadowDir, surfaceNormal) > 0 && LNL > 0)
            {
                const float3& emission = _GMaterials[lightD.material].emission;
                // https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
                float A = 0.5f * crLength;

                float SA = LNL * A * invShadowLength * invShadowLength;
                // writing this explicitly leads to NaNs
                //float lightPDF = 1.0f / (SA * DTriangleLights.size);
                state.light = state.mask * (NL * SA * DTriangleLights.size * BRDF * emission);


                // We invert the shadowrays to get coherent origins.
                Ray shadowRay(samplePoint + LNL * EPS * shadowDir + (1-LNL) * EPS * lightNormal, shadowDir, ray.pixeli);
                shadowRay.length = shadowLength - 2 * EPS;
                DShadowRayQueue.push(RayPacked(shadowRay));
            }
        }

        TriangleD* triangleD = _GVertexData + hitInfo.primitive_id;
        const float3 r = hitInfo.primitive_type == TRIANGLE
                ? SampleHemisphereCached(colliderNormal, randState, *triangleD, cache.cache_bucket_id)
                : SampleHemisphereCosine(colliderNormal, rand(randState), rand(randState));
        cullSecondary = dot(r, surfaceNormal) < 0 || dot(r, colliderNormal) < 0;
        const float f = fmaxf(dot(colliderNormal, r),0.0f);
        secondary = Ray(intersectionPos + EPS * f * r + EPS * (1 - f) * colliderNormal, r, ray.pixeli);
        if (hitInfo.primitive_type == TRIANGLE)
        {
            cache.sample_type = SAMPLE_BUCKET;
            cache.cum_mask = state.mask;
            cache.triangle_id = hitInfo.primitive_id;
        }
        state.mask *= PI * BRDF;
    }

    // Russian roulette
    float p = fminf(fmaxcompf(material.diffuse_color), 0.9f);
    if (!cullSecondary && rand(randState) < p)
    {
        state.mask = state.mask / p;
        DRayQueueNew.push(RayPacked(secondary));
    }
    else
    {
        cache.sample_type = SAMPLE_TERMINATE;
    } stateBuf.setState(ray.pixeli, state); 
    uint cache_id = ray.pixeli * MAX_RAY_DEPTH + bounce;
    sampleCache[cache_id] = cache;
}

// Traces the shadow rays
__global__ void kernel_connect(TraceStateSOA stateBuf)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DShadowRayQueue.size) return;

    const Ray& shadowRay = DShadowRayQueue[i].getRay();
    HitInfo hitInfo = traverseTopLevel<true>(shadowRay);
    if (hitInfo.intersected()) return;

    float3 light = get3f(stateBuf.lights[shadowRay.pixeli]);
    stateBuf.accucolors[shadowRay.pixeli] += make_float4(light,0.0f);
}

__global__ void kernel_add_to_screen(const TraceStateSOA stateBuf, cudaSurfaceObject_t texRef, RandState randState)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NR_PIXELS) return;
    const uint x = i % WINDOW_WIDTH;
    const uint y = i / WINDOW_WIDTH;

//    const float xf = (float)x / randState.blueNoiseSize.x;
 //   const float yf = (float)y / randState.blueNoiseSize.y;

    float3 color = get3f(stateBuf.accucolors[i]);
  //  color = make_float3(tex2D<float>(randState.blueNoise, xf, yf));
    float4 old_color_all;
    surf2Dread(&old_color_all, texRef, x*sizeof(float4), y);
    float3 old_color = fmaxf(make_float3(0.0f), get3f(old_color_all));
    surf2Dwrite(make_float4(old_color + color, old_color_all.w+1.0f), texRef, x*sizeof(float4), y);
}

__global__ void kernel_clear_screen(cudaSurfaceObject_t texRef)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);
    surf2Dwrite(make_float4(0), texRef, x*sizeof(float4), y);
}

__global__ void kernel_clear_rays()
{
    DRayQueue.clear();
    DShadowRayQueue.clear();
    DRayQueueNew.clear();
}

__global__ void kernel_swap_and_clear()
{
    swapc(DRayQueue, DRayQueueNew);
    DRayQueueNew.clear();
    DShadowRayQueue.clear();
}

#endif
