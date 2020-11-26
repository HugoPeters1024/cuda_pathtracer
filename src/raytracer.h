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
    Raytracer(SceneData sceneData, GLuint texture) : Application(sceneData, texture) {}
    virtual void Init() override;
    virtual void Draw(const Camera& camera, float currentTime, bool shouldClear) override;
};

void Raytracer::Init()
{
    // Assign the scene buffers to the global binding sites
    HTriangles = sceneData.h_vertex_buffer;
    HTriangleData = sceneData.h_data_buffer;
    HSpheres = HSizedBuffer<Sphere>(sceneData.h_sphere_buffer, sceneData.num_spheres);
    HPlanes = HSizedBuffer<Plane>(sceneData.h_plane_buffer, sceneData.num_planes);
    HMaterials = sceneData.h_material_buffer;
    HBVH = sceneData.h_bvh_buffer;

    screenBuffer = (float*)malloc(4 * NR_PIXELS * sizeof(float));
    omp_set_num_threads(8);
}

void Raytracer::Draw(const Camera& camera, float currentTime, bool shouldClear)
{
    max_depth = 5;
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

    glBindTexture(GL_TEXTURE_2D, texture);
    glTextureSubImage2D(texture, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_FLOAT, screenBuffer);
    glBindTexture(GL_TEXTURE_2D, 0);
}

float3 Raytracer::radiance(const Ray& ray, int iteration)
{
    if (iteration >= max_depth) return make_float3(0);
    HitInfo hitInfo = traverseBVHStack(ray, false);
    if (!hitInfo.intersected) return make_float3(0.2, 0.3, 0.6);

    float3 intersectionPos = ray.origin + (hitInfo.t - EPS) * ray.direction;

    const float3 originalNormal = getColliderNormal(hitInfo, intersectionPos);
    bool inside = dot(ray.direction, originalNormal) > 0;
    const float3 colliderNormal = inside ? -originalNormal : originalNormal;

    const Material& material = getColliderMaterial(hitInfo);
    float3 diffuse_color = make_float3(0);
    float3 refract_color = make_float3(0);
    float3 reflect_color = make_float3(0);

    float transmission = material.transmit;
    float reflect = material.reflect;
    float diffuse = 1 - transmission - reflect;

    if (diffuse > 0) {
        for (int i = 0; i < sceneData.num_point_lights; i++) {
            const PointLight &light = sceneData.h_point_light_buffer[i];
            float3 fromLight = intersectionPos - light.pos;
            // we occlude ourselves
            if (dot(fromLight, colliderNormal) >= 0) continue;

            float dis2light2 = dot(fromLight, fromLight);
            float dis2light = sqrt(dis2light2);
            fromLight /= dis2light;
            Ray shadowRay(light.pos + EPS * fromLight, fromLight, 0);
            shadowRay.length = dis2light - EPS;
            HitInfo shadowHit = traverseBVHStack(shadowRay, true);
            if (!shadowHit.intersected) {
                diffuse_color += light.color * dot(-fromLight, colliderNormal) / (dis2light2 * 3.1415926 * 4);
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
                refract_color = refract_color * make_float3(exp(-c.x * hitInfo.t), exp(-c.y *hitInfo.t), exp(-c.z * hitInfo.t));
            }
        }
    }

    if (reflect > 0)
    {
        Ray reflectRay = getReflectRay(ray, colliderNormal, intersectionPos);
        reflect_color = radiance(reflectRay, iteration+1);
    }

    return material.color * (diffuse * diffuse_color + transmission * refract_color + reflect * reflect_color);
}

#endif
