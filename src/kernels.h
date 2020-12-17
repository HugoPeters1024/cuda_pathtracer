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
    float u = atan2(normal.x, normal.z) / (2*3.1415926) + 0.5;
    float v = normal.y * 0.5 + 0.5;
    return make_float2(u,v);
}

HYBRID inline Ray transformRay(const Ray& ray, const glm::mat4x4& transform)
{
    glm::vec4 oldDir = glm::vec4(ray.direction.x, ray.direction.y, ray.direction.z, 0);
    glm::vec4 oldOrigin = glm::vec4(ray.origin.x, ray.origin.y, ray.origin.z, 1);

    glm::vec4 newDir = oldDir * transform;
    glm::vec4 newOrigin = oldOrigin * transform;

    Ray transformedRay = ray;
    transformedRay.direction = normalize(make_float3(newDir.x, newDir.y, newDir.z));
    transformedRay.origin = make_float3(newOrigin.x, newOrigin.y, newOrigin.z);
    return transformedRay;
}

HYBRID inline Material getColliderMaterial(const HitInfo& hitInfo, const Model* model)
{
    switch (hitInfo.primitive_type)
    {
        case TRIANGLE: return _GMaterials[model->trianglesD[hitInfo.primitive_id].material];
        case SPHERE:   return _GMaterials[_GSpheres[hitInfo.primitive_id].material];
        case PLANE:    return _GMaterials[_GPlanes[hitInfo.primitive_id].material];
    }
    assert(false);
}

HYBRID inline float3 getColliderNormal(const HitInfo& hitInfo, const float3& intersectionPoint, const Model* model)
{
    switch (hitInfo.primitive_type)
    {
        case TRIANGLE: {
            return model->trianglesD[hitInfo.primitive_id].n0;
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

HYBRID inline bool slabTest(const float3& rayOrigin, const float3& invRayDir, const Box& box, float& tmin)
{
    float3 t0 = (box.vmin - rayOrigin) * invRayDir;
    float3 t1 = (box.vmax - rayOrigin) * invRayDir;
    float3 tmin3 = fminf(t0,t1), tmax3 = fmaxf(t1,t0);
    tmin = fmaxcompf(tmin3);
    return fmincompf(tmax3) >= max(0.0f, tmin);
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

    // Precompute inv ray direction for better slab tests
    float3 invRayDir = 1.0f / ray.direction;

    // declare variables used in the loop
    float tnear, tfar;
    bool bnear, bfar;
    uint near_id, far_id;
    uint start, end;

    const Model model = _GModels[instance.model_id];
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
                if (rayTriangleIntersect(ray, model.trianglesV[i], t, u, v) && t < hitInfo.t)
                {
                    hitInfo.intersected = true;
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
    hitInfo.intersected = false;
    hitInfo.t = ray.length;
    hitInfo.primitive_id = 0;

    float t, u, v;

    for(int i=0; i<_GSpheres.size; i++)
    {
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
        if (rayPlaneIntersect(ray, _GPlanes[i], t) && t < hitInfo.t)
        {
            hitInfo.intersected = true;
            hitInfo.primitive_id = i;
            hitInfo.primitive_type = PLANE;
            hitInfo.t = t;
            if (anyIntersection) return hitInfo;
        }
    }

    // Any intersection is being used by shadow rays, but they can't intersect the light source
    for(int i=0;i<_GSphereLights.size; i++)
    {
        if (!anyIntersection && raySphereIntersect(ray, _GSphereLights[i], t) && t < hitInfo.t)
        {
            hitInfo.t = t;
            hitInfo.intersected = true;
            hitInfo.primitive_id = i;
            hitInfo.primitive_type = LIGHT;
        }
    }

    // TODO
    TopLevelBVH root = _GTopBVH[0];
    Instance rootInstance = _GInstances[root.leaf];
    Ray transformedRay = transformRay(ray, rootInstance.invTransform);
    traverseBVHStack(transformedRay, anyIntersection, hitInfo, rootInstance, root.leaf);
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
    return normalize(u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2));
}

__device__ float3 BRDFUniform(const float3& normal, uint& seed)
{
    float r1 = 2 * 3.1415926535 * rand(seed);
    float r2 = rand(seed);
    float r2s = sqrtf(r2);

    float3 w = normal;
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
    float3 v = cross(w,u);

    return normalize(u*r1*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2));
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

__device__ Ray getDiffuseRay(const Ray& ray, const float3 normal, const float3 intersectionPos, uint seed)
{
    float3 newDir = BRDF(normal, seed);
    return Ray(intersectionPos + EPS * newDir, newDir, ray.pixeli);
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
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DRayQueue.size) return;
    Ray ray = DRayQueue[i].getRay();
    TraceState state = stateBuf.getState(ray.pixeli);

    uint seed = getSeed(i,wang_hash(i),time);
    const HitInfo hitInfo = intersections[i].getHitInfo();
    if (!hitInfo.intersected) {
        // We consider the skydome a lightsource in the set of random bounces
        float2 uvCoords = normalToUv(ray.direction);
        float4 sk4 = tex2D<float4>(skydome, uvCoords.x, uvCoords.y);
        float3 sk = make_float3(sk4.x, sk4.y, sk4.z);
        //if (bounce > 0) sk = sk * 10;
        /*
        // Artificially increase contrast to make the sun a more apparent light source
        // without affecting the direct view of the image.
        if (bounce > 0)
        {
            // normalized brightness.
            float brightness = dot(sk, sk) / 3.0f;
            sk *= pow(brightness, 6) * 60;
        }
        */
        state.accucolor += state.mask * sk;
        stateBuf.setState(ray.pixeli, state);
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
                state.accucolor += state.mask * _GSphereLights[hitInfo.primitive_id].color;
                stateBuf.setState(ray.pixeli, state);
            }
        }
        else
        {
            state.accucolor += state.mask * _GSphereLights[hitInfo.primitive_id].color;
            stateBuf.setState(ray.pixeli, state);
        }
        return;
    }

    Instance* instance;
    Model* model;

    float3 intersectionPos = ray.origin + hitInfo.t * ray.direction;
    // Only triangles are always part of instances
    if (hitInfo.primitive_type == TRIANGLE)
    {
        instance = _GInstances + hitInfo.instance_id;
        model = _GModels + instance->model_id;
    }

    Material material = getColliderMaterial(hitInfo, model);
    float3 originalNormal = getColliderNormal(hitInfo, intersectionPos, model);

    // invert the normal and position transformation back to world space
    if (hitInfo.primitive_type == TRIANGLE)
    {
        glm::vec4 wn = glm::vec4(originalNormal.x, originalNormal.y, originalNormal.z, 0) * instance->transform;
        originalNormal = normalize(make_float3(wn.x, wn.y, wn.z));
    }

    bool inside = dot(ray.direction, originalNormal) > 0;
    float3 colliderNormal = inside ? -originalNormal : originalNormal;
    float3 surfaceNormal = colliderNormal;

    if (hitInfo.primitive_type == PLANE) {
        uint px = (uint)(fabs(intersectionPos.x/4));
        uint py = (uint)(fabs(intersectionPos.z/4));
        material.diffuse_color = (px + py)%2 == 0 ? make_float3(1) : make_float3(0.2);
    }

    // sample the texture of the material by redoing the intersection
    if (material.hasTexture || material.hasNormalMap)
    {
        const TriangleD& triangleData = model->trianglesD[hitInfo.primitive_id];
        const TriangleV& triangleV = model->trianglesV[hitInfo.primitive_id];

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
            float3 texNormal = normalize(get3f(texColor)*2-make_float3(1)) * make_float3(-1, 1, 1);
            texNormal = get3f(normalize(triangleData.TBN * make_float4(texNormal, 0)));
            if (dot(texNormal, colliderNormal) < 0) texNormal = -texNormal;
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
            secondary = getReflectRay(ray, colliderNormal, intersectionPos);
            // Color is only a diffuse color
            state.mask = state.mask * material.diffuse_color;
            if (dot(state.mask, state.mask) < 0.01) return;
        }
        float3 noiseDir = BRDF(secondary.direction, seed);
        secondary.direction = secondary.direction * (1 - material.glossy) + material.glossy * noiseDir;
    }
    else if (random - material.transmit < material.reflect)
    {
        state.fromSpecular = true;
        secondary = getReflectRay(ray, colliderNormal, intersectionPos);
        float3 noiseDir = BRDF(secondary.direction, seed);
        secondary.direction = secondary.direction * (1-material.glossy) + material.glossy * noiseDir;

        // Color is only a diffuse color
        state.mask = state.mask * material.diffuse_color;
        if (dot(state.mask, state.mask) < 0.01) return;
    }
    else 
    {
        // Color is only a diffuse color
        state.mask = state.mask * material.diffuse_color;
        if (dot(state.mask, state.mask) < 0.01) return;
        state.fromSpecular = false;

        secondary = getDiffuseRay(ray, colliderNormal, intersectionPos, seed);
        // due to normal mapping a sample might go into the surface
        cullSecondary = dot(surfaceNormal, secondary.direction) < 0;

        // Lambert term is not applied because it is implicit due to the cosine weights
        // for the new sampled direction.

        // Create a shadow ray for diffuse objects
        if (_NEE)
        {
            // Choose an area light at random
            uint lightSource = wang_hash(seed) % DSphereLights.size;
            const SphereLight& light = DSphereLights[lightSource];
            float3 fromLight = normalize(intersectionPos - light.pos);
            // Sample the hemisphere from that point.
            float3 r = BRDFUniform(fromLight, seed);

            // From the center of the light, go to sample point
            // (by definition of the BRDF on the visible by the origin (if not occluded)
            float3 samplePoint = light.pos + light.radius * r;

            float3 shadowDir = intersectionPos - samplePoint;
            float shadowLength = length(shadowDir);
            float invShadowLength = 1.0f / shadowLength;
            shadowDir *= invShadowLength;


            // otherwise we are our own occluder
            if (dot(colliderNormal, shadowDir) < 0)
            {
                float cost1 = dot(r, shadowDir);
                float cost2 = dot(colliderNormal, -shadowDir);
                float SA = (3.1415926 * light.radius * light.radius * cost1 * cost2) * invShadowLength * invShadowLength;

                state.light = state.mask * light.color * SA * DSphereLights.size;

                // contribution will not be worth it otherwise.
                if (dot(state.light, state.light) > 0.03)
                {
                    // We invert the shadowrays to get coherent origins.
                    Ray shadowRay(samplePoint + EPS * shadowDir, shadowDir, ray.pixeli);
                    shadowRay.length = shadowLength - 2 * EPS;
                    DShadowRayQueue.push(RayPacked(shadowRay));
                }
            }
        }
    }

    if (!cullSecondary) {
        // Russian roullete
        float p = clamp(fmax(fmax(material.diffuse_color.x, material.diffuse_color.y), material.diffuse_color.z), 0.1f, 1.0f);
        if (rand(seed) < p)
        {
            DRayQueueNew.push(RayPacked(secondary));
            state.mask *= (1.0f / p);
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
    if (hitInfo.intersected) return;

    float3 light = get3f(stateBuf.lights[shadowRay.pixeli]);
    stateBuf.accucolors[shadowRay.pixeli] += make_float4(light,0);
}

__global__ void kernel_add_to_screen(const TraceStateSOA stateBuf, cudaSurfaceObject_t texRef)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NR_PIXELS) return;
    const uint x = i % WINDOW_WIDTH;
    const uint y = i / WINDOW_WIDTH;

    TraceState state = stateBuf.getState(i);
    float3 color = state.accucolor;
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
