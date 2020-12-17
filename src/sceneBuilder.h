#ifndef H_SCENE_BUILDER
#define H_SCENE_BUILDER

#include "types.h"
#include "scene.h"

inline SceneData getOutsideScene()
{
    auto scene = new Scene();

    // Add all the materials
    Material white = Material::DIFFUSE(make_float3(0.4));
    auto whiteId = scene->addMaterial(white);

    Material cubeMat = Material::DIFFUSE(make_float3(1));
    cubeMat.transmit = 1.0f;
    cubeMat.refractive_index = 1.1;
    cubeMat.glossy = 0.02;
    cubeMat.absorption = make_float3(0.1, 0.5, 0.8);
    auto cubeMatId = scene->addMaterial(cubeMat);

    Material sibenikMat = Material::DIFFUSE(make_float3(0.8));
    auto sibenikMatId = scene->addMaterial(sibenikMat);

    Material teapotMat = Material::DIFFUSE(make_float3(1));
    teapotMat.reflect = 0.6;
    teapotMat.glossy = 0.08;
    auto teapotMatId = scene->addMaterial(teapotMat);

    Material lucyMat = Material::DIFFUSE(make_float3(0.5, 0.2, 0.3));
    lucyMat.transmit = 0.0f;
    lucyMat.refractive_index = 1.2;
    lucyMat.reflect = 0.0;
    lucyMat.glossy = 0.15;
    lucyMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto lucyMatId = scene->addMaterial(lucyMat);

    Material glassMat = Material::DIFFUSE(make_float3(1));
    glassMat.transmit = 1.0f;
    glassMat.refractive_index = 1.544;
    glassMat.glossy = 0.00f;
    glassMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto glassMatId = scene->addMaterial(glassMat);

    Material whiteGlass = Material::DIFFUSE(make_float3(1));
    whiteGlass.transmit = 1.0f;
    whiteGlass.refractive_index = 1.5;
    auto whiteGlassId = scene->addMaterial(whiteGlass);

    auto mirrorMat = Material::DIFFUSE(make_float3(1));
    mirrorMat.transmit = 0.0f;
    mirrorMat.refractive_index = 1.4f;
    mirrorMat.reflect = 1.0f;
    auto mirrorMatId = scene->addMaterial(mirrorMat);

    // Add all the objects
    //scene.addModel("sibenik.obj", 3, make_float3(0), make_float3(0,42,0), sibenikMatId);
    //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + (scene.triangles.size() * 3) / 4  );
  //  scene->addModel("cube.obj", 1, make_float3(0), make_float3(0), cubeMatId);
   //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1);
    //scene.addModel("teapot.obj", 1, make_float3(0), make_float3(-3,0,0), teapotMatId);
   // scene.addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,4.0), lucyMatId);
    uint house_model = scene->addModel("house.obj", 0.04, make_float3(0), make_float3(0), whiteId);

    glm::mat4x4 house_transform = glm::mat4x4(
            glm::vec4(1,0,0,15),
            glm::vec4(0,1,0,-2.5),
            glm::vec4(0,0,1,4),
            glm::vec4(0,0,0,1));

    house_transform = glm::rotate(house_transform, 1.8f, glm::vec3(0.0f,1.0f,0.0f));

    Instance house
    {
        house_model,
        house_transform,
        glm::inverse(house_transform),
    };

    scene->addInstance(house);

    scene->addPlane(Plane(make_float3(0,-1,0),-3, whiteId));

    scene->addSphere(Sphere(make_float3(0, 0, 0), 1, mirrorMatId));
    scene->addSphere(Sphere(make_float3(-2, -1, -3), 2, whiteGlassId));
    scene->addSphere(Sphere(make_float3(-2, -1, 3), 2, mirrorMatId));

    // Add the lights
    scene->addPointLight(PointLight(make_float3(-8,5,1), make_float3(150)));
    scene->addPointLight(PointLight(make_float3(-8,5,-5), make_float3(150, 0, 0)));
    scene->addPointLight(PointLight(make_float3(-8,5,5), make_float3(0, 150, 0)));

    scene->addSphereLight(SphereLight(make_float3(-8,5,0), 1, make_float3(150)));
    scene->addSphereLight(SphereLight(make_float3(-5,5,-5), 1, make_float3(150, 0, 0)));
    scene->addSphereLight(SphereLight(make_float3(-5,5,5), 1, make_float3(0, 150, 0)));
        
    return scene->finalize();
}

inline SceneData getSibenikScene()
{
    auto scene = new Scene();

    // Add all the materials
    Material white = Material::DIFFUSE(make_float3(0.4));
    auto whiteId = scene->addMaterial(white);

    Material cubeMat = Material::DIFFUSE(make_float3(1));
    cubeMat.transmit = 1.0f;
    cubeMat.refractive_index = 1.1;
    cubeMat.glossy = 0.02;
    cubeMat.absorption = make_float3(0.1, 0.5, 0.8);
    auto cubeMatId = scene->addMaterial(cubeMat);

    Material sibenikMat = Material::DIFFUSE(make_float3(0.7));
    auto sibenikMatId = scene->addMaterial(sibenikMat);

    Material teapotMat = Material::DIFFUSE(make_float3(1));
    teapotMat.reflect = 0.6;
    teapotMat.glossy = 0.08;
    auto teapotMatId = scene->addMaterial(teapotMat);

    Material lucyMat = Material::DIFFUSE(make_float3(0.98, 0.745, 0.02));
    lucyMat.reflect = 0.7;
    lucyMat.glossy = 0.08;
    auto lucyMatId = scene->addMaterial(lucyMat);

    Material glassMat = Material::DIFFUSE(make_float3(1));
    glassMat.transmit = 1.0f;
    glassMat.refractive_index = 1.544;
    glassMat.glossy = 0.00f;
    glassMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto glassMatId = scene->addMaterial(glassMat);

    Material whiteGlass = Material::DIFFUSE(make_float3(1));
    whiteGlass.transmit = 1.0f;
    whiteGlass.refractive_index = 1.5;
    auto whiteGlassId = scene->addMaterial(whiteGlass);

    auto mirrorMat = Material::DIFFUSE(make_float3(1));
    mirrorMat.transmit = 0.0f;
    mirrorMat.refractive_index = 1.4f;
    mirrorMat.reflect = 1.0f;
    auto mirrorMatId = scene->addMaterial(mirrorMat);

    // Add all the objects
    uint sibenikModel = scene->addModel("sibenik.obj", 1, make_float3(0), make_float3(0,0,0), sibenikMatId, true);

    glm::mat4x4 identity = glm::mat4x4(
            glm::vec4(1,0,0,0),
            glm::vec4(0,1,0,-12),
            glm::vec4(0,0,1,0),
            glm::vec4(0,0,0,1));


    Instance sibenikInstance =
    {
        sibenikModel,
        identity,
        inverse(identity),
    };

    scene->addInstance(sibenikInstance);

    scene->addModel("cube.obj", 1, make_float3(0.1, 0.3, 0.7), make_float3(-5, 1, 0), cubeMatId);
//    scene.addModel("cube_brian.obj", 1, make_float3(0), make_float3(-4, 6, 0), 0, true);
    //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + (scene.triangles.size() * 3) / 4  );
   // scene.addModel("cube.obj", 1, make_float3(0), make_float3(0), cubeMatId);
   //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1);
    //scene.addModel("teapot.obj", 1, make_float3(0), make_float3(-3,0,0), teapotMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,4.0), lucyMatId);
    /*
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,7.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,1.0), lucyMatId);

    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(0,0,4.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(0,0,7.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(0,0,1.0), lucyMatId);

    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-3,0,4.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-3,0,7.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-3,0,1.0), lucyMatId);
   // scene.addModel("house.obj", 0.04, make_float3(0), make_float3(15,-2.5,4), whiteId);

    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-12,0,4.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-12,0,7.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-12,0,1.0), lucyMatId);

    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-15,0,4.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-15,0,7.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-15,0,1.0), lucyMatId);

    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-18,0,4.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-18,0,7.0), lucyMatId);
    scene->addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(-18,0,1.0), lucyMatId);
     */

    scene->addPlane(Plane(make_float3(0,-1,0),-3, whiteId));

    //scene.addSphere(Sphere(make_float3(0, 0, 0), 1, mirrorMatId));
    scene->addSphere(Sphere(make_float3(-2, -1, -3), 2, whiteGlassId));
    scene->addSphere(Sphere(make_float3(-2, -1, 3), 2, mirrorMatId));

    // Add the lights
    scene->addPointLight(PointLight(make_float3(-8,5,1), make_float3(150)));
    scene->addSphereLight(SphereLight(make_float3(-8,5,0), 1, make_float3(60)));

    return scene->finalize();
}

inline SceneData getConferenceScene()
{
    auto scene = new Scene;
    Material white = Material::DIFFUSE(make_float3(0.4));
    auto whiteId = scene->addMaterial(white);

 //   scene->addModel("conference.obj", 0.2, make_float3(0), make_float3(0, 10, 0), whiteId);
    //scene->addModel("2Mtris.obj", 0.2, make_float3(0), make_float3(0, 10, 0), whiteId, false);
    uint model = scene->addModel("2Mtris.obj", 0.2, make_float3(0), make_float3(0, 0, 0), whiteId, true);

    glm::mat4x4 transform = glm::mat4x4(1.0f);
    transform = glm::translate(transform, glm::vec3(0, 10, 0));
    transform = glm::rotate(transform, 3.14152926f * 0.5f, glm::vec3(1.0f,0.0f,0.0f));

    Instance lostEmpire
    {
        model,
        transform,
        glm::inverse(transform),
    };
    
    scene->addInstance(lostEmpire);


    scene->addPointLight(PointLight(make_float3(-8,5,1), make_float3(150)));
    scene->addSphereLight(SphereLight(make_float3(0), 1, make_float3(150)));

    return scene->finalize();
}

#endif

