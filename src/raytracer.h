#ifndef H_RAYTRACER
#define H_RAYTRACER

#include <omp.h>
#include <math.h>
#include "types.h"
#include "constants.h"
#include "application.h"
#include "globals.h"
#include "kernels.h"

#ifndef _OPENMP
#warning Openmp not enabled!
#endif


class Raytracer : public Application
{
private:
    float* screenBuffer;
    uint max_depth;
    float3 radiance(const Ray& ray, int iteration = 0);

public:
    SceneBuffers sceneBuffers;
    Raytracer(Scene& scene, GLuint luminanceTexture, GLuint albedoTexture) 
        : Application(scene, luminanceTexture, albedoTexture) {}
    virtual void Init() override;
    virtual void Render(const Camera& camera, float currentTime, float frameTime, bool shouldClear) override;
    virtual void Finish() override {}
};

void Raytracer::Init()
{
    sceneBuffers.vertices = scene.allVertices.data();
    sceneBuffers.vertexData = scene.allVertexData.data();
    sceneBuffers.instances = scene.instances;
    sceneBuffers.models = scene.models.data();
    sceneBuffers.materials = scene.materials.data();
    sceneBuffers.topBvh = scene.topLevelBVH.data();
    sceneBuffers.spheres = scene.spheres.data();
    sceneBuffers.num_spheres = scene.spheres.size();
    sceneBuffers.planes = scene.planes.data();
    sceneBuffers.num_planes = scene.planes.size();

    screenBuffer = (float*)malloc(4 * NR_PIXELS * sizeof(float));
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif

    // Initialize the albedo texture to white because we won't use it anyway.
    for(uint i=0; i<NR_PIXELS*4; i++)
    {
        screenBuffer[i] = 1.0f;
    }

    glBindTexture(GL_TEXTURE_2D, albedoTexture);
    glTextureSubImage2D(luminanceTexture, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_FLOAT, screenBuffer);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Raytracer::Render(const Camera& camera, float currentTime, float frameTime, bool shouldClear)
{

    max_depth = shouldClear ? 2 : 7;
#pragma omp parallel for schedule (dynamic)
    for(uint i=0; i<NR_PIXELS; i++)
    {
        uint y = i / WINDOW_WIDTH;
        uint x = i % WINDOW_WIDTH;
        const Ray ray = camera.getRay(x, y);
        float3 color = radiance(ray);

        screenBuffer[x * 4 + y * 4 * WINDOW_WIDTH + 0] = color.x;
        screenBuffer[x * 4 + y * 4 * WINDOW_WIDTH + 1] = color.y;
        screenBuffer[x * 4 + y * 4 * WINDOW_WIDTH + 2] = color.z;
        screenBuffer[x * 4 + y * 4 * WINDOW_WIDTH + 3] = 1;
    }

    glBindTexture(GL_TEXTURE_2D, luminanceTexture);
    glTextureSubImage2D(luminanceTexture, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_FLOAT, screenBuffer);
    glBindTexture(GL_TEXTURE_2D, 0);
}

float3 Raytracer::radiance(const Ray& ray, int iteration)
{
    if (iteration >= max_depth) return make_float3(0);
    HitInfo hitInfo = traverseTopLevel<false>(sceneBuffers, ray);
    if (!hitInfo.intersected()) return make_float3(0.2, 0.3, 0.6);

    const Instance* instance = sceneBuffers.instances + hitInfo.instance_id;

    float3 intersectionPos = ray.origin + hitInfo.t * ray.direction;
    float3 originalNormal = getColliderNormal(sceneBuffers, hitInfo, intersectionPos);
    if (hitInfo.primitive_type == TRIANGLE)
    {
        originalNormal = normalize(instance->transform.mul(originalNormal, 0.0f));
    }

    bool inside = dot(ray.direction, originalNormal) > 0;
    const float3 colliderNormal = inside ? -originalNormal : originalNormal;

    const uint material_id = getColliderMaterialID(sceneBuffers, hitInfo, instance);
    Material material = sceneBuffers.materials[material_id];
    float3 diffuse_color = make_float3(0);
    float3 refract_color = make_float3(0);
    float3 reflect_color = make_float3(0);

    if (hitInfo.primitive_type == PLANE)
    {
        uint px = (uint)(fabsf(intersectionPos.x/4));
        uint py = (uint)(fabsf(intersectionPos.z/4));
        material.diffuse_color = (px + py)%2 == 0 ? make_float3(1) : make_float3(0.2);
    }

    float transmission = material.transmit;
    float reflect = material.reflect;
    float diffuse = 1 - transmission - reflect;

    if (diffuse > 0) {
        for (int i = 0; i < scene.pointLights.size(); i++) {
            const PointLight &light = scene.pointLights[i];
            float3 fromLight = intersectionPos - light.pos;
            // we occlude ourselves
            if (dot(fromLight, colliderNormal) >= 0) continue;

            float dis2light2 = dot(fromLight, fromLight);
            float dis2light = sqrtf(dis2light2);
            fromLight /= dis2light;
            Ray shadowRay(light.pos + EPS * fromLight, fromLight, 0);
            shadowRay.length = dis2light - 2 * EPS;
            HitInfo shadowHit = traverseTopLevel<true>(sceneBuffers, shadowRay);
            if (!shadowHit.intersected()) {
                diffuse_color += light.color * dot(-fromLight, colliderNormal) / dis2light2;
            }
        }
    }


    if (transmission > 0)
    {

       float changed_reflection = 0;
        Ray refractRay = getRefractRay(ray, colliderNormal, intersectionPos, material, inside, changed_reflection);
        transmission -= changed_reflection;
        reflect += changed_reflection;
        if (transmission > 0) {
            refract_color = radiance(refractRay, iteration+1);
            if (inside)
            {
                // Take away any absorpted light using Beer's law. when leaving the object
                float3 c = material.absorption;
                refract_color = refract_color * make_float3(expf(-c.x * hitInfo.t), expf(-c.y *hitInfo.t), expf(-c.z * hitInfo.t));
            }
        }
    }

    if (reflect > 0)
    {
        Ray reflectRay = getReflectRay(ray, colliderNormal, intersectionPos);
        reflect_color = radiance(reflectRay, iteration+1);
    }

    return material.diffuse_color * (diffuse * diffuse_color + transmission * refract_color + reflect * reflect_color);
}

#endif
