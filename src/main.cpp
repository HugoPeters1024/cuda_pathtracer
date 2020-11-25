
#include "constants.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

#include "types.h"
#include "gl_shader_utils.h"

#include "use_cuda.h"
#include "bvhBuilder.h"
#include "scene.h"
#include "globals.h"
#include "kernels.h"
#include "application.h"
#include "raytracer.h"
#include "pathtracer.h"
#include "keyboard.h"

static const char* quad_vs = R"(
#version 460

in vec2 pos;

out vec2 uv;

void main()
{
    gl_Position = vec4(pos, 0, 1);
    uv = (pos + vec2(1)) * 0.5;
}
)";

static const char* quad_fs = R"(
#version 460

in vec2 uv;
out vec4 color;

uniform sampler2D tex;
layout (location = 0) uniform float time;

void main() { 
    vec4 c = texture(tex, uv);
    color = vec4(c.xyz / c.w, 1);
    float gamma = 1.5;
    color.x = pow(color.x, 1.0f/gamma);
    color.y = pow(color.y, 1.0f/gamma);
    color.z = pow(color.z, 1.0f/gamma);
}
)";

void error_callback(int error, const char* description) { fprintf(stderr, "ERROR: %s/n", description); }
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


int main(int argc, char** argv) {
    if (!glfwInit()) return 2;

    glfwSetErrorCallback(error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Cuda Pathtracer", nullptr, nullptr);
    if (!window) return 3;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Compile the quad shader to display a texture
    GLuint quad_shader = GenerateProgram(CompileShader(GL_VERTEX_SHADER, quad_vs), CompileShader(GL_FRAGMENT_SHADER, quad_fs));
    float quad_vertices[12] = {
            -1.0, -1.0,
           1.0, -1.0,
           -1.0, 1.0,

           1.0, 1.0,
           -1.0, 1.0,
           1.0, -1.0,
    };

    GLuint quad_vao, quad_vbo;
    glGenVertexArrays(1, &quad_vao);
    glBindVertexArray(quad_vao);

    glGenBuffers(1, &quad_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    glEnableVertexArrayAttrib(quad_vao, 0);
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, nullptr);

    // Generate screen texture
    // list of GL formats that cuda supports: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1gd7be3ca8a7a739d57f0b558562c5706e
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);


    // Create a scene object
    Scene scene;

    // create materials
    Material white = Material::DIFFUSE(make_float3(1));
    auto whiteId = scene.addMaterial(white);

    Material cubeMat = Material::DIFFUSE(make_float3(1));
    cubeMat.transmit = 1.0f;
    cubeMat.refractive_index = 1.1;
    cubeMat.glossy = 0.02;
    cubeMat.absorption = make_float3(0.1, 0.5, 0.8);
    auto cubeMatId = scene.addMaterial(cubeMat);

    Material sibenikMat = Material::DIFFUSE(make_float3(1));
    auto sibenikMatId = scene.addMaterial(sibenikMat);

    Material lucyMat = Material::DIFFUSE(make_float3(0.822, 0.751, 0.412));
    lucyMat.transmit = 0.0f;
    lucyMat.reflect = 0.8;
    lucyMat.glossy = 0.09;
    lucyMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto lucyMatId = scene.addMaterial(lucyMat);

    Material glassMat = Material::DIFFUSE(make_float3(1));
    glassMat.transmit = 1.0f;
    glassMat.refractive_index = 1.544;
    glassMat.glossy = 0.00f;
    glassMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto glassMatId = scene.addMaterial(glassMat);

    auto mirrorMat = Material::DIFFUSE(make_float3(1));
    mirrorMat.transmit = 0.0f;
    mirrorMat.refractive_index = 1.4f;
    mirrorMat.reflect = 1.0f;
    auto mirrorMatId = scene.addMaterial(mirrorMat);

    //scene.addModel("cube.obj", 1, make_float3(0), make_float3(0), cubeMatId);
   //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1);
    //scene.addModel("sibenik.obj", 1, make_float3(0), make_float3(0,12,0), sibenikMatId);
    scene.addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,4.0), lucyMatId);
    //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1300);
    //
    scene.addPlane(Plane(make_float3(0,-1,0),-4, whiteId));

    scene.addSphere(Sphere(make_float3(0, 2, 1), 1, glassMatId));
    scene.addSphere(Sphere(make_float3(0, 0, 0), 1, mirrorMatId));

    scene.addPointLight(PointLight(make_float3(-8,12,1), make_float3(90)));
        
    printf("Generating a BVH using the SAH heuristic, this might take a moment...\n");
    SceneData sceneData = scene.finalize();

    bool PATHRACER = true;

    // Create the application
    Pathtracer pathtracerApp = Pathtracer(sceneData, texture);
    Raytracer raytracerApp = Raytracer(sceneData, texture);


    // add a sphere as light source
    Sphere light(make_float3(-8,5,1), 0.05, -1);
    float3 lightColor = make_float3(150);


    // Set the initial camera values;
    Camera camera(make_float3(0,2,-3), make_float3(0,0,1), 1.5);
    double runningAverageFps = 0;
    int tick = 0;

    printf("BVHNode is %i bytes\n", sizeof(BVHNode));
    printf("Triangle is %i bytes\n", sizeof(BVHNode));
    printf("Ray is %i bytes\n", sizeof(Ray));
    printf("HitInfo is %i bytes\n", sizeof(HitInfo));
    printf("TraceState is %i bytes\n", sizeof(TraceState));

    pathtracerApp.Init();
    raytracerApp.Init();

    Keyboard keyboard(window);

    bool shouldClear = true;
    while (!glfwWindowShouldClose(window))
    {
        tick++;
        double start = glfwGetTime();

        cudaSafe( cudaMemcpyToSymbol(DLight, &light, sizeof(Sphere)) );
        cudaSafe( cudaMemcpyToSymbol(DLight_Color, &lightColor, sizeof(float3)) );

        if (PATHRACER)
            pathtracerApp.Draw(camera, glfwGetTime(), shouldClear);
        else
            raytracerApp.Draw(camera, glfwGetTime(), shouldClear);


        // Draw the texture
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(quad_shader);
        glUniform1f(0, glfwGetTime());
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(quad_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Handle IO and swap the backbuffer
        camera.update(window);
        shouldClear = camera.hasMoved();
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) { 
            lightColor *= 0.97; 
            sceneData.h_point_light_buffer[0].color *= 0.97;
            shouldClear = true;
        }
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) { 
            lightColor *= 1.03; 
            sceneData.h_point_light_buffer[0].color *= 1.03;
            shouldClear = true;
        }
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) { light.radius *= 1.03; shouldClear = true;}
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) { light.radius *= 0.97; shouldClear = true;}

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { light.pos.y += 0.02; shouldClear = true;}
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { light.pos.y -= 0.02; shouldClear = true;}
        if (keyboard.isPressed(SWITCH)) { PATHRACER = !PATHRACER; shouldClear = true; }
        glfwPollEvents();
        glfwSwapBuffers(window);
        keyboard.swapBuffers();

        double fps = 1.0f / (glfwGetTime() - start);
        runningAverageFps = runningAverageFps * 0.95 + 0.05 * fps;
        if (tick % 60 == 0) printf("running average fps: %f\n", runningAverageFps);

        // Vsync is broken in GLFW for my card, so just hack it in.
        while (glfwGetTime() - start < 1.0 / 60.0) {}
    }

    glfwDestroyWindow(window);
    return 0;
}
