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

HYBRID inline float2 normalToUv(const float3& normal)
{
    float u = atan2(normal.x, normal.z) / (2.0f*PI) + 0.5f;
    float v = normal.y * 0.5f + 0.5f;
    return make_float2(u,v);
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

HYBRID inline Material getColliderMaterial(const HitInfo& hitInfo)
{
    switch (hitInfo.primitive_type)
    {
        case TRIANGLE: return _GMaterials[_GVertexData[hitInfo.primitive_id].material];
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
    if (fabs(a) < 0.001) return false;
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
    if (fabs(q)<EPS) return false;
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
    return fmincompf(tmax3) >= fmax(0.0f, tmin);
#else
    __m128 bmin = _mm_setr_ps(box.vmin.x, box.vmin.y, box.vmin.z, 0.0f);
    __m128 bmax = _mm_setr_ps(box.vmax.x, box.vmax.y, box.vmax.z, 0.0f);
    __m128 sdRayOrigin = _mm_setr_ps(rayOrigin.x, rayOrigin.y, rayOrigin.z, 0.0f);
    __m128 sdInvRayDir = _mm_setr_ps(invRayDir.x, invRayDir.y, invRayDir.z, 0.0f);
    __m128 t0 = _mm_mul_ps(_mm_sub_ps(bmin, sdRayOrigin), sdInvRayDir);
    __m128 t1 = _mm_mul_ps(_mm_sub_ps(bmax, sdRayOrigin), sdInvRayDir);
    __m128 sdtmin = _mm_min_ps(t0, t1);
    __m128 sdtmax = _mm_max_ps(t0, t1);
    tmin = fmax(fmax(sdtmin[0], sdtmin[1]), sdtmin[2]);
    float tmax = fmin(fmin(sdtmax[0], sdtmax[1]), sdtmax[2]);
    return tmax >= fmax(0.0f, tmin);
#endif
}

HYBRID bool rayTriangleIntersect(const Ray& ray, const TriangleV& triangle, float& t, float& u, float& v)
{
    float3 v0v1 = triangle.v1 - triangle.v0;
    float3 v0v2 = triangle.v2 - triangle.v0;
    float3 pvec = cross(ray.direction, v0v2);
    float det = dot(v0v1, pvec);
    float invDet = 1.0f / det;

    float3 tvec = ray.origin - triangle.v0;
    u = dot(tvec, pvec) * invDet;

    float3 qvec = cross(tvec, v0v1);
    v = dot(ray.direction, qvec) * invDet;

    t = dot(v0v2, qvec) * invDet;
    return fabs(det) > 0.0001f && (u >= 0 && u <= 1) && (v >= 0 && u + v <= 1) && t > 0;
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

HYBRID void traverseBVHStack(const Ray& ray, bool anyIntersection, HitInfo& hitInfo, const Instance& instance, uint instanceIdx)
{
    float t, u, v;

    const uint STACK_SIZE = 18;
    uint stack[STACK_SIZE];
    uint size = 0;

    // declare variables used in the loop
    float tnear, tfar;
    bool bnear, bfar;
    uint near_id, far_id;
    uint start, end;

    const float3 invRayDir = 1.0f / ray.direction;

    const Model& model = _GModels[instance.model_id];
    BVHNode current = model.bvh[0];
    if (boxtest(current.getBox(), ray.origin, invRayDir, tnear, hitInfo)) stack[size++] = 0;

    while(size > 0)
    {
        current = model.bvh[stack[--size]];

        if (current.isLeaf())
        {
            start = current.t_start();
            end = start + current.t_count();
            for(uint i=start; i<end; i++)
            {
                if (rayTriangleIntersect(ray, _GVertices[i], t, u, v) && t < hitInfo.t)
                {
                    hitInfo.primitive_id = i;
                    hitInfo.primitive_type = TRIANGLE;
                    hitInfo.t = t;
                    hitInfo.instance_id = instanceIdx;
                    if (anyIntersection) return;
                }
            }
        }
        else
        {
            near_id = current.child1();
            far_id = current.child1() + 1;
            bnear = boxtest(model.bvh[near_id].getBox(), ray.origin, invRayDir, tnear, hitInfo);
            bfar = boxtest(model.bvh[far_id].getBox(), ray.origin, invRayDir, tfar, hitInfo);

            // push on the stack, first the far child
            if (bfar) stack[size++] = far_id;
            if (bnear)  stack[size++] = near_id;

            if (bnear && bfar && tnear > tfar) {
                swapc(stack[size-1], stack[size-2]);
            }

            //assert (size < STACK_SIZE);
        }
    }
}

HYBRID HitInfo traverseTopLevel(const Ray& ray, bool anyIntersection)
{
    HitInfo hitInfo;
    hitInfo.primitive_id = 0xffffffff;
    hitInfo.t = ray.length;

    float t, tmin;

    for(int i=0; i<_GSpheres.size; i++)
    {
        if (raySphereIntersect(ray, _GSpheres[i], t) && t < hitInfo.t)
        {
            hitInfo.primitive_id = i;
            hitInfo.primitive_type = SPHERE;
            hitInfo.t = t;
            if (anyIntersection) return hitInfo;
        }
    }

    for(int i=0; i<_GPlanes.size; i++)
    {
        if (rayPlaneIntersect(ray, _GPlanes[i], t) && t < hitInfo.t)
        {
            hitInfo.primitive_id = i;
            hitInfo.primitive_type = PLANE;
            hitInfo.t = t;
            if (anyIntersection) return hitInfo;
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
            traverseBVHStack(transformedRay, anyIntersection, hitInfo, instance, current.leaf);
            if (anyIntersection && hitInfo.intersected()) return hitInfo;
        }
        else
        {
            stack[size++] = current.child1;
            stack[size++] = current.child2;
        }
    }

    return hitInfo;
}

__device__ float3 SampleHemisphere(const float3& normal, uint& seed)
{
    float r0 = rand(seed), r1 = rand(seed);
    float r = sqrtf(r0);
    float theta = 2 * 3.1415926535 * r1;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float3 sample =  make_float3( x, y, sqrt(1 - r0));

    const float3& w = normal;
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
    float3 v = normalize(cross(w,u));

    return normalize(make_float3(
            dot(sample, make_float3(u.x, v.x, w.x)),
            dot(sample, make_float3(u.y, v.y, w.y)),
            dot(sample, make_float3(u.z, v.z, w.z))));
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
    }

    float3 refractDir = normalize(eta * ray.direction + normal * (eta * costi - sqrt(k)));

    // fresnell equation for reflection contribution
    float sinti = sqrt(max(0.0f, 1.0f - costi - costi));
    float costt = sqrt(1 - eta * eta * sinti * sinti);
    float spol = (n1 * costi - n2 * costt) / (n1 * costi + n2 * costt);
    float ppol = (n1 * costt - n2 * costi) / (n1 * costt + n2 * costi);

    reflected = 0.5 * (spol * spol + ppol * ppol);
    return Ray(intersectionPos + EPS * refractDir, refractDir, ray.pixeli);
}

__device__ Ray getDiffuseRay(const Ray& ray, const float3& normal, const float3& intersectionPos, uint& seed)
{
    float3 newDir = SampleHemisphere(normal, seed);
    float f = dot(normal, newDir);
    return Ray(intersectionPos + EPS * f * newDir + EPS * (1-f) * normal, newDir, ray.pixeli);
}


__global__ void kernel_clear_state(TraceStateSOA state)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NR_PIXELS) return;
    state.accucolors[i] = make_float4(0.0f);
    state.masks[i] = make_float4(1.0f,1.0f,1.0f,__int_as_float(1));
}

__global__ void kernel_generate_primary_rays(Camera camera, float time)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_LIMIT(x,y);
    uint seed = getSeed(x,y,time);
    RayPacked ray = RayPacked(camera.getRay(x,y,seed));
    DRayQueue.push(ray);
}

__global__ void kernel_extend(HitInfoPacked* intersections, uint bounce)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= DRayQueue.size) return;
    const Ray ray = DRayQueue[i].getRay();
    HitInfo hitInfo = traverseTopLevel(ray, false);
    intersections[i] = HitInfoPacked(hitInfo);
}


__global__ void kernel_shade(const HitInfoPacked* intersections, TraceStateSOA stateBuf, float time, int bounce, cudaTextureObject_t skydome)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DRayQueue.size) return;
    Ray ray = DRayQueue[i].getRay();
    TraceState state = stateBuf.getState(ray.pixeli);

    uint x = ray.pixeli % WINDOW_WIDTH;
    uint y = ray.pixeli / WINDOW_WIDTH;
    uint seed = getSeed(x,y,time);
    const HitInfo hitInfo = intersections[i].getHitInfo();
    if (!hitInfo.intersected()) {
        // We consider the skydome a lightsource in the set of random bounces
        float2 uvCoords = normalToUv(ray.direction);
        float4 sk4 = tex2D<float4>(skydome, uvCoords.x, uvCoords.y);
        float3 sk = make_float3(sk4.x, sk4.y, sk4.z);

        state.accucolor += state.mask * sk;
        stateBuf.setState(ray.pixeli, state);
        return;
    }

    // Only triangles are always part of instances but that is the
    // responsibility of the code dereferencing the pointer.
    Instance* instance = _GInstances + hitInfo.instance_id;
    float3 intersectionPos = ray.origin + hitInfo.t * ray.direction;
    Material material = getColliderMaterial(hitInfo);
    float3 surfaceNormal = getColliderNormal(hitInfo, intersectionPos);

    // invert the normal and position transformation back to world space
    if (hitInfo.primitive_type == TRIANGLE)
    {
        glm::vec4 wn = instance->transform * glm::vec4(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z, 0);
        surfaceNormal = normalize(make_float3(wn.x, wn.y, wn.z));
    }

    bool inside = dot(ray.direction, surfaceNormal) > 0;
    float3 colliderNormal = inside ? -surfaceNormal : surfaceNormal;
    surfaceNormal = colliderNormal;

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
        uint px = (uint)(fabs(intersectionPos.x/4 + 1000));
        uint py = (uint)(fabs(intersectionPos.z/4 + 1000));
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
            float4 texColor = tex2D<float4>(material.texture, uv.x, uv.y);
            // According to the mtl spec texture should be multiplied by the diffuse color
            // https://www.loc.gov/preservation/digital/formats/fdd/fdd000508.shtml
            material.diffuse_color = material.diffuse_color * make_float3(texColor.x, texColor.y, texColor.z);
        }

        // sample the normal of the material by redoing the intersection
        if (material.hasNormalMap)
        {
            float4 texColor = tex2D<float4>(material.normal_texture, uv.x, uv.y);
            float3 texNormalT = normalize(get3f((texColor)*2)-make_float3(1));
            float3 texNormal = normalize(make_float3(
                    dot(texNormalT, make_float3(triangleData.tangent.x, triangleData.bitangent.x, triangleData.normal.x)),
                    dot(texNormalT, make_float3(triangleData.tangent.y, triangleData.bitangent.y, triangleData.normal.y)),
                    dot(texNormalT, make_float3(triangleData.tangent.z, triangleData.bitangent.z, triangleData.normal.z))
            ));

            // Transform the normal from model space to world space
            glm::vec4 wn = instance->transform * glm::vec4(texNormal.x, texNormal.y, texNormal.z, 0.0f);
            texNormal = normalize(make_float3(wn.x, wn.y, wn.z));
            if (dot(texNormal, colliderNormal) < 0.0f) texNormal = -texNormal;
            colliderNormal = texNormal;
        }
    }

    // Create a secondary ray either diffuse or reflected
    Ray secondary;
    bool cullSecondary = false;
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
            state.mask *= material.diffuse_color;
            secondary = getReflectRay(ray, colliderNormal, intersectionPos);
        }
        float3 noiseDir = SampleHemisphere(secondary.direction, seed);
        secondary.direction = secondary.direction * (1.0f - material.glossy) + material.glossy * noiseDir;
    }
    else if (random - material.transmit < material.reflect)
    {
        state.fromSpecular = true;
        state.mask *= material.diffuse_color;
        secondary = getReflectRay(ray, colliderNormal, intersectionPos);
        float3 noiseDir = SampleHemisphere(secondary.direction, seed);
        secondary.direction = secondary.direction * (1.0f-material.glossy) + material.glossy * noiseDir;
    }
    else 
    {
        state.fromSpecular = false;
        const float3 BRDF = material.diffuse_color / PI;

        // Create a shadow ray for diffuse objects
        if (_NEE)
        {
            // Choose an area light at random
            seed = wang_hash(seed);
            uint lightSource = seed % DTriangleLights.size;
            const TriangleLight& light = DTriangleLights[lightSource];
            const TriangleV& lightV = _GVertices[light.triangle_index];
            const TriangleD& lightD = _GVertexData[light.triangle_index];
            const Instance& lightInstance = _GInstances[light.instance_index];

            // transform the vertices to world space
            const float3 v0 = get3f(lightInstance.transform * glm::vec4(lightV.v0.x, lightV.v0.y, lightV.v0.z, 1));
            const float3 v1 = get3f(lightInstance.transform * glm::vec4(lightV.v1.x, lightV.v1.y, lightV.v1.z, 1));
            const float3 v2 = get3f(lightInstance.transform * glm::vec4(lightV.v2.x, lightV.v2.y, lightV.v2.z, 1));

            const float3 v0v1 = v1 - v0;
            const float3 v0v2 = v2 - v0;
            const float3 cr = cross(v0v1, v0v2);
            const float crLength = length(cr);

            float u=rand(seed), v=rand(seed);
            if (u+v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
            float3 samplePoint = v0 + u * v0v1 + v * v0v2;

            float3 shadowDir = intersectionPos - samplePoint;
            float shadowLength = length(shadowDir);
            float invShadowLength = 1.0f / shadowLength;
            shadowDir *= invShadowLength;

            float3 lightNormal = cr / crLength;
            lightNormal = dot(lightNormal, shadowDir) < 0 ? -lightNormal : lightNormal;

            float NL = dot(colliderNormal, -shadowDir);
            float LNL = dot(lightNormal, shadowDir);

            // otherwise we are our own occluder or view the backface of the light
            if (NL > 0 && dot(surfaceNormal, -shadowDir) > 0 && LNL > 0)
            {
                const float3& emission = _GMaterials[lightD.material].emission;
                // https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
                float A = 0.5f * crLength;

                float SA = LNL * A * invShadowLength * invShadowLength;
                // writing this explicitly leads to NaNs
                //float lightPDF = 1.0f / (SA * DTriangleLights.size);
                state.light = state.mask * (NL * SA * DTriangleLights.size * BRDF * emission);


                // We invert the shadowrays to get coherent origins.
                Ray shadowRay(samplePoint + NL * EPS * shadowDir + (1-NL) * EPS * lightNormal, shadowDir, ray.pixeli);
                shadowRay.length = shadowLength - 2 * EPS;
                DShadowRayQueue.push(RayPacked(shadowRay));
            }
        }
        secondary = getDiffuseRay(ray, colliderNormal, intersectionPos, seed);
        // due to normal mapping a sample might go into the surface
        cullSecondary = dot(surfaceNormal, secondary.direction) <= 0;

        // Writing this explicitly leads to NaNs
        //float NL = dot(colliderNormal, secondary.direction);
        //float PDF = NL / PI;

        state.mask = state.mask * (PI * BRDF);
    }

    if (!cullSecondary) {
        // Russian roulette
        float p = clamp(fmaxcompf(material.diffuse_color), 0.1f, 0.9f);
        if (rand(seed) <= p)
        {
            state.mask = state.mask / p;
            DRayQueueNew.push(RayPacked(secondary));
        }
    }
    stateBuf.setState(ray.pixeli, state);
}

// Traces the shadow rays
__global__ void kernel_connect(TraceStateSOA stateBuf)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DShadowRayQueue.size) return;

    const Ray& shadowRay = DShadowRayQueue[i].getRay();
    HitInfo hitInfo = traverseTopLevel(shadowRay, true);
    if (hitInfo.intersected()) return;

    float3 light = get3f(stateBuf.lights[shadowRay.pixeli]);
    stateBuf.accucolors[shadowRay.pixeli] += make_float4(light,0.0f);
}

__global__ void kernel_add_to_screen(const TraceStateSOA stateBuf, cudaSurfaceObject_t texRef)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NR_PIXELS) return;
    const uint x = i % WINDOW_WIDTH;
    const uint y = i / WINDOW_WIDTH;

    float3 color = get3f(stateBuf.accucolors[i]);
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
