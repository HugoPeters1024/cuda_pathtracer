
#include "constants.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "cxxopts.hpp"

#include "types.h"
#include "gl_shader_utils.h"

#include "use_cuda.h"
#include "bvhBuilder.h"
#include "scene.h"
#include "sceneBuilder.h"
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
    // slightly zoom in to hide artificats
    // from chrommatic abberation at the edges
    uv = (pos * 0.97 + vec2(1)) * 0.5;
}
)";

static const char* quad_fs = R"(
#version 460

in vec2 uv;
out vec4 color;

uniform sampler2D tex;
layout (location = 0) uniform float time;

void main() { 
    vec2 fromCenter = uv - vec2(0.5);
    vec4 sampleR = texture(tex, uv + 0.011 * fromCenter);
    vec4 sampleG = texture(tex, uv + 0.007 * fromCenter);
    vec4 sampleB = texture(tex, uv + 0.003 * fromCenter);
    float gamma = 1.8f;
    color.x = pow(sampleR.x / sampleR.w, 1.0f/gamma);
    color.y = pow(sampleG.y / sampleG.w, 1.0f/gamma);
    color.z = pow(sampleB.z / sampleB.w, 1.0f/gamma);

    // vignetting
    color *= 1 - dot(fromCenter, fromCenter);
}
)";

void error_callback(int error, const char* description) { fprintf(stderr, "ERROR: %s/n", description); }
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

int main(int argc, char** argv) {
    cxxopts::Options options("AVGR 2020-2021 by Hugo Peters", "Raytracer/Pathtracer demo program");

    options.add_options()
        ("s,scene", "Scene to run", cxxopts::value<std::string>()->default_value("outside"))
    ;

    auto cmdArgs = options.parse(argc, argv);

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
    //float* screenBuf = (float*)malloc(NR_PIXELS * 4 * sizeof(float));
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);


    const char* sceneName = cmdArgs["scene"].as<std::string>().c_str();
    printf("Loading scene '%s', this might take a moment\n", sceneName);
    Scene scene = getScene(sceneName);

    bool PATHRACER = false;

    // Create the applications
    Pathtracer pathtracerApp = Pathtracer(scene, texture);
    Raytracer raytracerApp = Raytracer(scene, texture);


    // Set the initial camera values;
    Camera camera(make_float3(0,2,-3), make_float3(0,0,1), 1.5);
    double runningAverageFps = 0;
    int tick = 0;

    // Show off
    printf("BVHNode is %lu bytes\n", sizeof(BVHNode));
    printf("Triangle is %lu bytes\n", sizeof(BVHNode));
    printf("Ray is %lu bytes\n", sizeof(Ray));
    printf("HitInfo is %lu bytes\n", sizeof(HitInfo));
    printf("TraceState is %lu bytes\n", sizeof(TraceState));

    pathtracerApp.Init();
    raytracerApp.Init();

    // Input utility
    Keyboard keyboard(window);

    bool shouldClear = true;
    float frameTime;
    while (!glfwWindowShouldClose(window))
    {
        tick++;
        float start = glfwGetTime();

        if (PATHRACER)
            pathtracerApp.Render(camera, glfwGetTime(), frameTime, shouldClear);
        else
            raytracerApp.Render(camera, glfwGetTime(), frameTime, shouldClear);

        // update while possibly asynchronous rendering is going on
        scene.update(glfwGetTime());

        if (PATHRACER)
            pathtracerApp.Finish();
        else
            raytracerApp.Finish();

        // Verify the texture to check for outliers
        /*
        glBindTexture(GL_TEXTURE_2D, texture);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, screenBuf);
        glBindTexture(GL_TEXTURE_2D, 0);
        float maxv = 0;
        for(int i=0; i<NR_PIXELS; i++)
        {
            maxv = std::max(screenBuf[i] / tick, maxv);
        }

        printf("brightest pixel: %f\n", maxv);
        */


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
            scene.sphereLights[0].color *= 0.97; 
            scene.pointLights[0].color *= 0.97;
            shouldClear = true;
        }
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) { 
            scene.sphereLights[0].color *= 1.03;
            scene.pointLights[0].color *= 1.03;
            shouldClear = true;
        }
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) { scene.sphereLights[0].radius *= 1.03; shouldClear = true;}
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) { scene.sphereLights[0].radius *= 0.97; shouldClear = true;}

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { scene.sphereLights[0].pos.y += 0.02; shouldClear = true;}
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { scene.sphereLights[0].pos.y -= 0.02; shouldClear = true;}
        if (keyboard.isPressed(SWITCH_MODE)) { PATHRACER = !PATHRACER; shouldClear = true; }
        if (keyboard.isPressed(SWITCH_NEE)) { HNEE = !HNEE; shouldClear = true; }
        glfwPollEvents();
        glfwSwapBuffers(window);
        keyboard.swapBuffers();

        double fps = 1.0f / (glfwGetTime() - start);
        runningAverageFps = runningAverageFps * 0.95 + 0.05 * fps;
        if (tick % 60 == 0) printf("running average fps: %f\n", runningAverageFps);

        // Vsync is broken in GLFW for my card, so just hack it in.
        frameTime = glfwGetTime() - start;
        if (frameTime < 1.0 / 60.0) {
            std::this_thread::sleep_for(std::chrono::milliseconds((uint)(((1.0/60.0f)-frameTime)*1000)));
        }
    }

    glfwDestroyWindow(window);
    return 0;
}
