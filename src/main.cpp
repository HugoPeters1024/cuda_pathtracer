#include <driver_types.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

#define GL_GLEXT_PROTOTYPES 1
#define GL3_PROTOTYPES      1
#include <GLFW/glfw3.h>
#include "gl_shader_utils.h"

#include "use_cuda.h"
#include "types.h"
#include "bvhBuilder.h"
#include "globals.h"
#include "kernels.h"


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
    color = texture(tex, uv);
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
    float* screen = (float*)malloc(WINDOW_WIDTH * WINDOW_HEIGHT * 4 * sizeof(float));
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, screen);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register the texture with cuda
    cudaGraphicsResource* pGraphicsResource;
    cudaSafe( cudaGraphicsGLRegisterImage(&pGraphicsResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
    cudaArray *arrayPtr;

    Sphere spheres[2] {
        Sphere {
            make_float3(-0.4,1.0,0.5),
            0.34 
        },
        Sphere {
            make_float3(0.3,1.5,0),
            0.7
        }
    };
    Sphere* sphereBuf;
    cudaSafe( cudaMalloc(&sphereBuf, sizeof(spheres)) );
    cudaSafe( cudaMemcpy(sphereBuf, spheres, sizeof(spheres), cudaMemcpyHostToDevice) );

    Box boxes[1] {
        Box {
            make_float3(-0.5, 2, -0.5),
            make_float3(0.5, 3, 0.5),
        }
    };
    Box* boxBuf;
    cudaSafe( cudaMalloc(&boxBuf, sizeof(boxes)) );
    cudaSafe( cudaMemcpy(boxBuf, boxes, sizeof(boxes), cudaMemcpyHostToDevice) );

    Scene scene;
    scene.addModel("teapot.obj", make_float3(1), 1, make_float3(0), 0);
    scene.addModel("cube.obj", make_float3(0.8,0.2,0.2), 8, make_float3(0), 0.6);
   // scene.addModel("sibenik.obj", make_float3(1), 1, make_float3(0,8,0), 0);
    //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1300);
    printf("Generating a BVH using the SAH heuristic at a depth of 3, this might take a moment...\n");
    BVHTree* bvh = scene.finalize();

    std::vector<Triangle> newTriangles;
    std::vector<BVHNode> newBvh;
    sequentializeBvh(bvh, newTriangles, newBvh);

    assert(newBvh.size() == bvh->treeSize());

    Triangle* triangleBuf;
    cudaSafe( cudaMalloc(&triangleBuf, newTriangles.size() * sizeof(Triangle)) );
    cudaSafe( cudaMemcpy(triangleBuf, &newTriangles[0], newTriangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice) );

    BVHNode* bvhBuf;
    printf("BVH Size: %ul\n", newBvh.size());
    cudaSafe( cudaMalloc(&bvhBuf, newBvh.size() * sizeof(BVHNode)) );
    cudaSafe( cudaMemcpy(bvhBuf, &newBvh[0], newBvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice) );

    Ray* rayBuf;
    cudaSafe ( cudaMalloc(&rayBuf, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(Ray)) );

    // Set the global bvh buffer pointer
    // We set this globally instead of kernel parameter, otherwise we would have
    // to pass the whole array constantly to small functions like sibling.
    cudaSafe( cudaMemcpyToSymbol(GBVH, &bvhBuf, sizeof(bvhBuf)) );

    // Do the same for the triangle buffer
    cudaSafe( cudaMemcpyToSymbol(GTriangles, &triangleBuf, sizeof(triangleBuf)) );

    // Set the initial camera values;
    Camera camera(make_float3(0,2,-3), make_float3(0,0,1), 1);
    double runningAverageFps = 0;
    int tick = 0;

    printf("BVHNode is %i bytes\n", sizeof(BVHNode));
    printf("Triangle is %i bytes\n", sizeof(BVHNode));

    while (!glfwWindowShouldClose(window))
    {
        tick++;
        double start = glfwGetTime();

        // Unbind the texture from OpenGL

        // Map the resource to Cuda
        cudaSafe ( cudaGraphicsMapResources(1, &pGraphicsResource) );

        // Get a pointer to the memory
        cudaSafe ( cudaGraphicsSubResourceGetMappedArray(&arrayPtr, pGraphicsResource, 0, 0) );

        // Wrap the cudaArray in a surface object
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = arrayPtr;
        cudaSurfaceObject_t inputSurfObj = 0;
        cudaSafe ( cudaCreateSurfaceObject(&inputSurfObj, &resDesc) );

        // Calculate the thread size and warp size
        int tx = 8;
        int ty = 8;
        dim3 dimBlock(WINDOW_WIDTH/tx+1, WINDOW_HEIGHT/ty+1);
        dim3 dimThreads(tx,ty);

        float time = glfwGetTime();
        cudaSafe( cudaMemcpyToSymbol(GTime, &time, sizeof(float)) );
//        kernel_create_primary_rays<<<dimBlock, dimThreads>>>(rayBuf, camera);
        kernel_pathtracer<<<dimBlock, dimThreads>>>(rayBuf, inputSurfObj, glfwGetTime(), camera);
        kernel_shadows<<<dimBlock, dimThreads>>>(rayBuf, inputSurfObj);
        cudaSafe ( cudaDeviceSynchronize() );

        // Unmap the resource from cuda
        cudaSafe ( cudaGraphicsUnmapResources(1, &pGraphicsResource) );

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
        glfwPollEvents();
        glfwSwapBuffers(window);

        double fps = 1.0f / (glfwGetTime() - start);
        runningAverageFps = runningAverageFps * 0.95 + 0.05 * fps;
        if (tick % 60 == 0) printf("running average fps: %f\n", runningAverageFps);

        // Vsync is broken in GLFW for my card, so just hack it in.
        while (glfwGetTime() - start < 1.0 / 60.0) {}
    }

    glfwDestroyWindow(window);
    return 0;
}
