#include <stdio.h>
#include <iostream>
#include <chrono>

#define GL_GLEXT_PROTOTYPES 1
#define GL3_PROTOTYPES      1
#include <GLFW/glfw3.h>
#include "gl_shader_utils.h"

#include <driver_types.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>
#include <cuda_surface_types.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <texture_types.h>


#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

struct mandelstate
{
    double centerx;
    double centery;
    double scale;
};


__device__ float mandel_iterate(double x, double y)
{
    const int max_iteration = 512;
    int i =0;
    double a = 0, b = 0, w = 0, x2 = 0, y2 =0;
    while(i++ < max_iteration && x2 + y2 <= 4) {
        a = x2 - y2 + x;
        b = w - x2 - y2 + y;
        x2 = a * a;
        y2 = b * b;
        w = (a + b) * (a + b);
    }
    return i / (float)max_iteration;
}

__global__ void mandelbrot(cudaSurfaceObject_t texRef, mandelstate state) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WINDOW_WIDTH && y < WINDOW_HEIGHT) {
        double sx = ((x / (double)WINDOW_WIDTH) - 0.5);
        double sy = ((y / (double)WINDOW_HEIGHT) - 0.5);

        double fx = sx / state.scale - state.centerx;
        double fy = sy / state.scale - state.centery;
        float m = mandel_iterate(fx, fy);
        surf2Dwrite(m, texRef, x*sizeof(float4), y);
    }
}

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

// level is [0,5], assumed to be a whole number
vec3 rainbow(float level)
{
	/*
		Target colors
		=============
		
		L  x   color
		0  0.0 vec4(1.0, 0.0, 0.0, 1.0);
		1  0.2 vec4(1.0, 0.5, 0.0, 1.0);
		2  0.4 vec4(1.0, 1.0, 0.0, 1.0);
		3  0.6 vec4(0.0, 0.5, 0.0, 1.0);
		4  0.8 vec4(0.0, 0.0, 1.0, 1.0);
		5  1.0 vec4(0.5, 0.0, 0.5, 1.0);
	*/
	
	float r = float(level <= 2.0) + float(level > 4.0) * 0.5;
	float g = max(1.0 - abs(level - 2.0) * 0.5, 0.0);
	float b = (1.0 - (level - 4.0) * 0.5) * float(level >= 4.0);
	return vec3(r, g, b);
}

vec3 smoothRainbow (float x)
{
    float level1 = floor(x*6.0);
    float level2 = min(6.0, floor(x*6.0) + 1.0);
    
    vec3 a = rainbow(level1);
    vec3 b = rainbow(level2);
    
    return mix(a, b, fract(x*6.0));
}

void main() { 
    float m = texture(tex, uv).r;
    color = vec4(smoothRainbow(m),1);
}
)";

void error_callback(int error, const char* description) { fprintf(stderr, "ERROR: %s/n", description); }
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

double centerx = 0, centery = 0, scale = 1;
double mx, my;

int main(int argc, char** argv) {
    if (!glfwInit()) return 2;

    glfwSetErrorCallback(error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Cuda Pathtracer", nullptr, nullptr);
    if (!window) return 3;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetScrollCallback(window, scroll_callback);

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
    for(int i=0; i<WINDOW_WIDTH*WINDOW_HEIGHT*4; i++) screen[i] = 0.4f;
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_FLOAT, screen);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Register the texture with cuda
    cudaGraphicsResource* pGraphicsResource;
    cudaSafe( cudaGraphicsGLRegisterImage(&pGraphicsResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard) );
    cudaArray *arrayPtr;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of cuda devices: %i\n", deviceCount);

    glfwGetCursorPos(window, &mx, &my);

    int tick = 0;
    while (!glfwWindowShouldClose(window))
    {
        double start = glfwGetTime();
        tick++;
        // Unbind the texture from OpenGL
        glBindTexture(GL_TEXTURE_2D, 0);

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
        dim3 dimBlock(16,16);
        dim3 dimGrid((WINDOW_WIDTH  + dimBlock.x - 1) / dimBlock.x,
                     (WINDOW_HEIGHT + dimBlock.y - 1) / dimBlock.y);


        // Call the kernel
        mandelstate state { centerx, centery, scale };
        mandelbrot<<<dimGrid,dimBlock>>>(inputSurfObj, state);
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

        // Handle IO and swap the backbuffer
        glfwPollEvents();
        glfwSwapBuffers(window);
        printf("fps: %f\n", 1.0f / (glfwGetTime() - start));

        // Update the center of the screen.
        double new_mx, new_my;
        glfwGetCursorPos(window, &new_mx, &new_my);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {

            centerx += 0.001 * (new_mx - mx) / scale;
            centery -= 0.001 * (new_my - my) / scale;

        }
        mx = new_mx;
        my = new_my;

        // Vsync is broken in GLFW for my card, so just hack it in.
        while (glfwGetTime() - start < 1.0 / 60.0) {}
    }

    glfwDestroyWindow(window);
    return 0;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (yoffset == 0) return;

    double aspectx = (double)WINDOW_WIDTH / WINDOW_HEIGHT;
    double aspecty = 1 / aspectx;

    // Calculate the current coords under the cursor
    double fx = ((mx / (double)WINDOW_WIDTH) - 0.5) / scale - centerx;
    double fy = ((my / (double)WINDOW_HEIGHT) - 0.5) / scale - centery;

    // Zoom
    scale *= (1 + yoffset / 10);

    // Calulate the difference between old and new
    double tx = centerx - ((mx / (double)WINDOW_WIDTH - 0.5) / scale - fx);
    double ty = centery - ((my / (double)WINDOW_HEIGHT - 0.5) / scale - fy);

    centerx -= tx;
    centery += ty;
}
