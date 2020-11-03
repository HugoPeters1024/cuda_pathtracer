#include <stdio.h>
#include <iostream>
#include <chrono>

#define GL_GLEXT_PROTOTYPES 1
#define GL3_PROTOTYPES      1
#include <GLFW/glfw3.h>
#include "gl_shader_utils.h"

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

#include "use_cuda.h"
#include "types.h"
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
    cudaSafe( cudaGraphicsGLRegisterImage(&pGraphicsResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard) );
    cudaArray *arrayPtr;

    Sphere* sphereBuf;
    cudaSafe (cudaMalloc(&sphereBuf, 1*sizeof(Sphere)));
    Sphere sphere1 {
        make_float3(0,1.5,0),
        1,
    };
    cudaMemcpy(sphereBuf, &sphere1, 1*sizeof(Sphere), cudaMemcpyHostToDevice);

    float3 test = make_float3(1,1,1);
    float3 ntest = normalize(test);
    printf("test vec: %f, %f, %f\n", ntest.x, ntest.y, ntest.z);

    while (!glfwWindowShouldClose(window))
    {
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
        dim3 dimBlock(32,32);
        dim3 dimGrid((WINDOW_WIDTH  + dimBlock.x - 1) / dimBlock.x,
                     (WINDOW_HEIGHT + dimBlock.y - 1) / dimBlock.y);


        kernel_pathtracer<<<dimGrid, dimBlock>>>(inputSurfObj, sphereBuf, glfwGetTime());
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
        glfwPollEvents();
        glfwSwapBuffers(window);

        // Vsync is broken in GLFW for my card, so just hack it in.
//        printf("theoretical fps: %f\n", 1.0f / (glfwGetTime() - start));
        while (glfwGetTime() - start < 1.0 / 60.0) {}
    }

    glfwDestroyWindow(window);
    return 0;
}
