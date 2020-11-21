
#include "constants.h"
#include <cuda_runtime_api.h>
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
#include "scene.h"
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
    vec4 c = texture(tex, uv);
    color = vec4(c.xyz / c.w, 1);
    float gamma = 1.8;
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

    // Create a scene object
    Scene scene;

    // create materials
    Material cubeMat = Material::DIFFUSE(make_float3(1));
    cubeMat.transmit = 1.0f;
    cubeMat.refractive_index = 1.1;
    cubeMat.glossy = 0.02;
    cubeMat.absorption = make_float3(0.1, 0.5, 0.8);
    auto cubeMatId = scene.addMaterial(cubeMat);

    Material sibenikMat = Material::DIFFUSE(make_float3(1));
    auto sibenikMatId = scene.addMaterial(sibenikMat);

    Material lucyMat = Material::DIFFUSE(make_float3(0.822, 0.751, 0.412));
    lucyMat.absorption = make_float3(1-0.722, 1-0.451, 1-0.012) * 5;
    lucyMat.transmit = 0.6f;
    lucyMat.reflect = 0.6f;
    lucyMat.refractive_index = 1.1;
    auto lucyMatId = scene.addMaterial(lucyMat);

    Material glassMat = Material::DIFFUSE(make_float3(1));
    glassMat.transmit = 1.0f;
    glassMat.refractive_index = 1.125f;
    glassMat.glossy = 0.05f;
    glassMat.absorption = make_float3(0.01, 0.4, 0.4);
    auto glassMatId = scene.addMaterial(glassMat);

    auto mirrorMat = Material::DIFFUSE(make_float3(1));
    mirrorMat.reflect = 1.0f;
    auto mirrorMatId = scene.addMaterial(mirrorMat);

    scene.addModel("cube.obj", 1, make_float3(0), make_float3(0), cubeMatId);
  //  scene.addModel("cube.obj", 0.5, make_float3(0), make_float3(0), cubeMat2);
    //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1);
    scene.addModel("sibenik.obj", 1, make_float3(0), make_float3(0,12,0), sibenikMatId);
    scene.addModel("lucy.obj",  0.005, make_float3(-3.1415926/2,0,3.1415926/2), make_float3(3,0,4.0), lucyMatId);
    //scene.triangles = std::vector<Triangle>(scene.triangles.begin(), scene.triangles.begin() + 1300);

    printf("Generating a BVH using the SAH heuristic, this might take a moment...\n");
    SceneData sceneData = scene.finalize();

    // Upload the host buffers to cuda
    TriangleV* d_vertex_buffer;
    cudaSafe( cudaMalloc(&d_vertex_buffer, sceneData.num_triangles * sizeof(TriangleV)) );
    cudaSafe( cudaMemcpy(d_vertex_buffer, sceneData.h_vertex_buffer, sceneData.num_triangles * sizeof(TriangleV), cudaMemcpyHostToDevice) );

    TriangleD* d_data_buffer;
    cudaSafe( cudaMalloc(&d_data_buffer, sceneData.num_triangles * sizeof(TriangleD)) );
    cudaSafe( cudaMemcpy(d_data_buffer, sceneData.h_data_buffer, sceneData.num_triangles * sizeof(TriangleD), cudaMemcpyHostToDevice) );

    BVHNode* d_bvh_buffer;
    cudaSafe( cudaMalloc(&d_bvh_buffer, sceneData.num_bvh_nodes * sizeof(BVHNode)) );
    cudaSafe( cudaMemcpy(d_bvh_buffer, sceneData.h_bvh_buffer, sceneData.num_bvh_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice) );

    // Assign to the global binding sites
    cudaSafe( cudaMemcpyToSymbol(GTriangles, &d_vertex_buffer, sizeof(d_vertex_buffer)) );
    cudaSafe( cudaMemcpyToSymbol(GTriangleData, &d_data_buffer, sizeof(d_data_buffer)) );
    cudaSafe( cudaMemcpyToSymbol(GBVH, &d_bvh_buffer, sizeof(d_bvh_buffer)) );

    // add a sphere as light source
    Sphere light(make_float3(-4,-1,1), 0.05, -1);
    float3 lightColor = make_float3(150);

    Sphere spheres[2] = {
            Sphere(make_float3(-8, 2, 1), 1, glassMatId),
            Sphere(make_float3(0, 0, 0), 1, mirrorMatId),
    };
    SizedBuffer<Sphere>(spheres, 2, GSpheres);

    // Send materials to gpu
    Material* matBuf;
    cudaSafe( cudaMalloc(&matBuf, scene.materials.size() * sizeof(Material)));
    cudaSafe( cudaMemcpy(matBuf, &scene.materials[0], scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice) );
    cudaSafe( cudaMemcpyToSymbol(GMaterials, &matBuf, sizeof(matBuf)) );



    Ray* rayBuf;
    cudaSafe ( cudaMalloc(&rayBuf, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(Ray)) );



    // queue of rays used in wavefront tracing
    AtomicQueue<Ray> rayQueue(NR_PIXELS);
    AtomicQueue<Ray> shadowRayQueue(NR_PIXELS);
    AtomicQueue<Ray> rayQueueNew(NR_PIXELS);


    HitInfo* intersectionBuf;
    cudaSafe( cudaMalloc(&intersectionBuf, NR_PIXELS * sizeof(HitInfo)) );

    TraceState* traceBuf;
    cudaSafe( cudaMalloc(&traceBuf, NR_PIXELS * sizeof(TraceState)) );

    // Set the initial camera values;
    Camera camera(make_float3(0,2,-3), make_float3(0,0,1), 1.5);
    double runningAverageFps = 0;
    int tick = 0;

    printf("BVHNode is %i bytes\n", sizeof(BVHNode));
    printf("Triangle is %i bytes\n", sizeof(BVHNode));
    printf("Ray is %i bytes\n", sizeof(Ray));
    printf("HitInfo is %i bytes\n", sizeof(HitInfo));
    printf("TraceState is %i bytes\n", sizeof(TraceState));

    bool shouldClear = false;
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
        int tx = 32;
        int ty = 32;
        dim3 dimBlock(WINDOW_WIDTH/tx+1, WINDOW_HEIGHT/ty+1);
        dim3 dimThreads(tx,ty);

        float time = glfwGetTime();
        cudaSafe( cudaMemcpyToSymbol(GTime, &time, sizeof(float)) );
        cudaSafe( cudaMemcpyToSymbol(GLight, &light, sizeof(Sphere)) );
        cudaSafe( cudaMemcpyToSymbol(GLight_Color, &lightColor, sizeof(float3)) );

        // clear the ray queue and update on gpu
        if (shouldClear)
            kernel_clear_screen<<<dimBlock, dimThreads>>>(inputSurfObj);


        kernel_clear_state<<<NR_PIXELS/1024, 1024>>>(traceBuf);

        // Generate primary rays in the ray queue
        rayQueue.clear();
        rayQueue.syncToDevice(GRayQueue);
        kernel_generate_primary_rays<<<dimBlock, dimThreads>>>(camera, glfwGetTime());
        rayQueue.syncFromDevice(GRayQueue);
        assert (rayQueue.size == WINDOW_WIDTH * WINDOW_HEIGHT);


        uint max_bounces = shouldClear ? 1 : 8;
        for(int bounces = 0; bounces < max_bounces; bounces++) {

            // Test for intersections with each of the rays,
            kernel_extend<<<rayQueue.size / 64 + 1, 64>>>(intersectionBuf, rayQueue.size);

            // Foreach intersection, possibly create shadow rays and secondary rays.
            shadowRayQueue.clear();
            shadowRayQueue.syncToDevice(GShadowRayQueue);
            rayQueueNew.clear();
            rayQueueNew.syncToDevice(GRayQueueNew);
            kernel_shade<<<rayQueue.size / 1024 + 1, 1024>>>(intersectionBuf, rayQueue.size, traceBuf, glfwGetTime());
            shadowRayQueue.syncFromDevice(GShadowRayQueue);
            rayQueueNew.syncFromDevice(GRayQueueNew);

            // Sample the light source for every shadow ray
            kernel_connect<<<shadowRayQueue.size / 256 + 1, 256>>>(shadowRayQueue.size, traceBuf);

            // swap the ray buffers
            rayQueueNew.syncToDevice(GRayQueue);
            std::swap(rayQueue, rayQueueNew);
        }

        // Write the final state accumulator into the texture
        kernel_add_to_screen<<<NR_PIXELS / 1024 + 1, 1024>>>(traceBuf, inputSurfObj);


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
        shouldClear = camera.hasMoved();
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) { lightColor *= 0.97; shouldClear = true;}
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) { lightColor *= 1.03; shouldClear = true;}
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) { light.radius *= 1.03; shouldClear = true;}
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) { light.radius *= 0.97; shouldClear = true;}
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
