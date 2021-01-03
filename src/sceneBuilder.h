#ifndef H_SCENE_BUILDER
#define H_SCENE_BUILDER

#include "types.h"
#include "scene.h"

inline Scene getOutsideScene()
{
    auto scene = Scene();

    // Add all the materials
    Material white = Material::DIFFUSE(make_float3(0.4));
    auto whiteId = scene.addMaterial(white);

    Material cubeMat = Material::DIFFUSE(make_float3(1));
    cubeMat.transmit = 1.0f;
    cubeMat.refractive_index = 1.1;
    cubeMat.glossy = 0.02;
    cubeMat.absorption = make_float3(0.1, 0.5, 0.8);
    auto cubeMatId = scene.addMaterial(cubeMat);

    Material sibenikMat = Material::DIFFUSE(make_float3(0.8));
    auto sibenikMatId = scene.addMaterial(sibenikMat);

    Material teapotMat = Material::DIFFUSE(make_float3(1));
    teapotMat.reflect = 0.6;
    teapotMat.glossy = 0.08;
    auto teapotMatId = scene.addMaterial(teapotMat);

    Material lucyMat = Material::DIFFUSE(make_float3(0.5, 0.2, 0.3));
    lucyMat.transmit = 0.0f;
    lucyMat.refractive_index = 1.2;
    lucyMat.reflect = 0.0;
    lucyMat.glossy = 0.15;
    lucyMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto lucyMatId = scene.addMaterial(lucyMat);

    Material glassMat = Material::DIFFUSE(make_float3(1));
    glassMat.transmit = 1.0f;
    glassMat.refractive_index = 1.544;
    glassMat.glossy = 0.00f;
    glassMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto glassMatId = scene.addMaterial(glassMat);

    Material whiteGlass = Material::DIFFUSE(make_float3(1));
    whiteGlass.transmit = 1.0f;
    whiteGlass.refractive_index = 1.5;
    auto whiteGlassId = scene.addMaterial(whiteGlass);

    auto mirrorMat = Material::DIFFUSE(make_float3(1));
    mirrorMat.transmit = 0.0f;
    mirrorMat.refractive_index = 1.4f;
    mirrorMat.reflect = 1.0f;
    auto mirrorMatId = scene.addMaterial(mirrorMat);

    // Add all the objects
    //scene.addModel("sibenik.obj", 3, make_float3(0), make_float3(0,42,0), sibenikMatId);
    //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + (scene.triangles.size() * 3) / 4  );
  //  scene->addModel("cube.obj", 1, make_float3(0), make_float3(0), cubeMatId);
   //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1);
    //scene.addModel("teapot.obj", 1, make_float3(0), make_float3(-3,0,0), teapotMatId);
   // scene.addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,4.0), lucyMatId);
    //uint house_model = scene->addModel("house.obj", 0.04, make_float3(0), make_float3(0), whiteId);
    //GameObject house(house_model);
    //house.position = make_float3(-15, -2.5, 4);
    //scene->addObject(house);

    uint cubeModel = scene.addModel("cube.obj", 1, make_float3(0), make_float3(0), cubeMatId);
    for(int i=0; i<10; i++)
    {
        GameObject cube(cubeModel);
        cube.kind = 1;
        cube.position.x = 10 * sin(i * 2 * 3.1415926) ;
        cube.position.z = 10 * cos(i * 2 * 3.1415926) ;
        cube.rotation.x = i * 3.1415926;
        scene.addObject(cube);
    }

    scene.addHandler([](Scene& scene, const Keyboard& keyboard, float t){
            float f = 0;
            for(int i=0; i<scene.objects.size(); i++)
            {
                GameObject& obj = scene.objects[i];
                if (obj.kind !=1) continue;
                obj.position.x = 10 * sin(f + t/10.0f) ;
                obj.position.z = 10 * cos(f + t/10.0f) ;
                obj.rotation.x = f;
                f += 2 * 0.3141592;
            }
            scene.invalidate();
    });


    scene.addPlane(Plane(make_float3(0,-1,0),-3, whiteId));

//    scene->addSphere(Sphere(make_float3(0, 0, 0), 1, mirrorMatId));
 //   scene->addSphere(Sphere(make_float3(-2, -1, -3), 2, whiteGlassId));
  //  scene->addSphere(Sphere(make_float3(-2, -1, 3), 2, mirrorMatId));

    // Add the lights
    scene.addPointLight(PointLight(make_float3(-8,5,1), make_float3(50)));
    scene.addPointLight(PointLight(make_float3(-8,5,-5), make_float3(50, 0, 0)));
    scene.addPointLight(PointLight(make_float3(-8,5,5), make_float3(0, 50, 0)));

        
    scene.finalize();
    return scene;
}

inline Scene getSibenikScene()
{
    auto scene = Scene();

    // Add all the materials
    Material white = Material::DIFFUSE(make_float3(0.4));
    auto whiteId = scene.addMaterial(white);

    Material cubeMat = Material::DIFFUSE(make_float3(1));
    cubeMat.transmit = 1.0f;
    cubeMat.refractive_index = 1.1;
    cubeMat.glossy = 0.02;
    cubeMat.absorption = make_float3(0.1, 0.5, 0.8);
    cubeMat.emission = make_float3(8);
    auto cubeMatId = scene.addMaterial(cubeMat);

    Material sibenikMat = Material::DIFFUSE(make_float3(0.2));
    auto sibenikMatId = scene.addMaterial(sibenikMat);

    Material teapotMat = Material::DIFFUSE(make_float3(1));
    teapotMat.reflect = 0.6;
    teapotMat.glossy = 0.08;
    auto teapotMatId = scene.addMaterial(teapotMat);

    Material lucyMat = Material::DIFFUSE(make_float3(0.98, 0.745, 0.02));
    lucyMat.reflect = 0.7;
    lucyMat.glossy = 0.08;
    auto lucyMatId = scene.addMaterial(lucyMat);

    Material glassMat = Material::DIFFUSE(make_float3(1));
    glassMat.transmit = 1.0f;
    glassMat.refractive_index = 1.544;
    glassMat.glossy = 0.00f;
    glassMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto glassMatId = scene.addMaterial(glassMat);

    Material whiteGlass = Material::DIFFUSE(make_float3(1));
    whiteGlass.transmit = 1.0f;
    whiteGlass.refractive_index = 1.5;
    auto whiteGlassId = scene.addMaterial(whiteGlass);

    auto mirrorMat = Material::DIFFUSE(make_float3(1));
    mirrorMat.transmit = 0.0f;
    mirrorMat.refractive_index = 1.4f;
    mirrorMat.reflect = 1.0f;
    auto mirrorMatId = scene.addMaterial(mirrorMat);

    // Add all the objects
    uint sibenikModel = scene.addModel("sibenik.obj", 1, make_float3(0), make_float3(0,0,0), sibenikMatId, true);

    GameObject sibenikObj(sibenikModel);
    sibenikObj.position.y = 12;
    scene.addObject(sibenikObj);

    uint cubeModel = scene.addModel("cube.obj", 1, make_float3(0), make_float3(0), cubeMatId);
    GameObject cubeObj(cubeModel);
    cubeObj.position = make_float3(0);
    cubeObj.kind = 5;
    scene.addObject(cubeObj);

    uint sphereModel = scene.addModel("sphere.obj", 50, make_float3(0), make_float3(0), cubeMatId);
    GameObject sphereObj(sphereModel);
    sphereObj.scale = make_float3(1);
    //scene.addObject(sphereObj);

    uint lucyModel = scene.addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,4.0), lucyMatId);
    GameObject lucyObj(lucyModel);
    scene.addObject(lucyObj);

   // scene.addPlane(Plane(make_float3(0,1,0),3, whiteId));


    //scene.addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,4.0), lucyMatId);

    //scene.addSphere(Sphere(make_float3(0, 0, 0), 1, mirrorMatId));
    scene.addSphere(Sphere(make_float3(-2, -1, -3), 2, whiteGlassId));
    scene.addSphere(Sphere(make_float3(-2, -1, 3), 2, mirrorMatId));

    // Add the lights
    scene.addPointLight(PointLight(make_float3(-8,5,1), make_float3(150)));

    scene.finalize();
    return scene;
}

inline const Scene getMinecraftScene()
{
    auto scene = Scene();
    Material white = Material::DIFFUSE(make_float3(0.4));
    auto whiteId = scene.addMaterial(white);

 //   scene->addModel("conference.obj", 0.2, make_float3(0), make_float3(0, 10, 0), whiteId);
    //scene->addModel("2Mtris.obj", 0.2, make_float3(0), make_float3(0, 10, 0), whiteId, false);
    uint model = scene.addModel("vokselia_spawn.obj", 20.0f, make_float3(0), make_float3(0, 0, 0), whiteId, true);

    GameObject obj(model);
//    obj.scale=make_float3(5.0f);
    scene.addObject(obj);


    scene.addPointLight(PointLight(make_float3(-8,5,1), make_float3(150)));

    scene.finalize();
    return scene;
}

inline const Scene get2MillionScene()
{
    auto scene = Scene();
    Material white = Material::DIFFUSE(make_float3(0.4));
    auto whiteId = scene.addMaterial(white);

 //   scene->addModel("conference.obj", 0.2, make_float3(0), make_float3(0, 10, 0), whiteId);
    //scene->addModel("2Mtris.obj", 0.2, make_float3(0), make_float3(0, 10, 0), whiteId, false);
    uint model = scene.addModel("2Mtris.obj", 0.2f, make_float3(0), make_float3(0, 0, 0), whiteId, false);

    GameObject obj(model);
    obj.rotation.x = - 3.1415926535 / 2;
//    obj.scale=make_float3(5.0f);
    scene.addObject(obj);


    scene.addPointLight(PointLight(make_float3(-8,5,1), make_float3(150)));

    scene.finalize();
    return scene;
}

inline Scene getScene(const char* sceneName)
{
    if (strcmp(sceneName, "outside") == 0)
        return getOutsideScene();
    if (strcmp(sceneName, "sibenik") == 0)
        return getSibenikScene();
    if (strcmp(sceneName, "minecraft") == 0)
        return getMinecraftScene();
    if (strcmp(sceneName, "2mtris") == 0)
        return get2MillionScene();

    printf("Scene '%s' does not exist!\n", sceneName);
    exit(-5);
}

#endif

