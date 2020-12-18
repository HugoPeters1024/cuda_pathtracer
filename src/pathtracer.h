#ifndef H_PATHTRACER
#define H_PATHTRACER

#include "application.h"
#include "globals.h"
#include "kernels.h"
#include <driver_types.h>

class Pathtracer : public Application
{
private:
    cudaGraphicsResource* pGraphicsResource;
    cudaArray *arrayPtr;
    AtomicQueue<RayPacked> rayQueue;
    AtomicQueue<RayPacked> shadowRayQueue;
    AtomicQueue<RayPacked> rayQueueNew;
    DSizedBuffer<Sphere> dSphereBuffer;
    DSizedBuffer<Plane> dPlaneBuffer;
    DSizedBuffer<SphereLight> dSphereLightBuffer;
    HitInfoPacked* intersectionBuf;
    TraceStateSOA traceBufSOA;
    cudaTextureObject_t dSkydomeTex;
    Instance* h_instances;
    Instance* d_instances;
    TopLevelBVH* d_topBvh;

public:
    Pathtracer(SceneData& sceneData, GLuint texture) : Application(sceneData, texture) {}

    virtual void Init() override;
    virtual void Draw(const Camera& camera, float currentTime, bool shouldClear) override;
};

void Pathtracer::Init()
{
    dSkydomeTex = loadTextureHDR("skydome.hdr");

    // Register the texture with cuda as preperation for interop.
    cudaSafe( cudaGraphicsGLRegisterImage(&pGraphicsResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore) );

    dSphereBuffer = DSizedBuffer<Sphere>(sceneData.h_sphere_buffer, sceneData.num_spheres, &DSpheres);
    dPlaneBuffer = DSizedBuffer<Plane>(sceneData.h_plane_buffer, sceneData.num_planes, &DPlanes);
    dSphereLightBuffer = DSizedBuffer<SphereLight>(sceneData.h_sphere_light_buffer, sceneData.num_sphere_lights, &DSphereLights);


    // Host buffer of device buffers... ;)
    Model hd_models[sceneData.num_models];

    // copy over the model but set the gpu buffers as source
    memcpy(hd_models, sceneData.h_models, sceneData.num_models * sizeof(Model));

    for(int i=0; i<sceneData.num_models; i++)
    {
        TriangleV* d_vertex_buffer;
        TriangleD* d_data_buffer;
        BVHNode* d_bvh_buffer;

        // Upload the host buffers to cuda
        cudaSafe( cudaMalloc(&d_vertex_buffer, sceneData.h_models[i].nrTriangles * sizeof(TriangleV)) );
        cudaSafe( cudaMalloc(&d_data_buffer, sceneData.h_models[i].nrTriangles * sizeof(TriangleD)) );
        cudaSafe( cudaMalloc(&d_bvh_buffer, sceneData.h_models[i].nrBvhNodes * sizeof(TriangleD)) );

        cudaSafe( cudaMemcpy(d_vertex_buffer, sceneData.h_models[i].trianglesV, sceneData.h_models[i].nrTriangles * sizeof(TriangleV), cudaMemcpyHostToDevice) );
        cudaSafe( cudaMemcpy(d_data_buffer, sceneData.h_models[i].trianglesD, sceneData.h_models[i].nrTriangles * sizeof(TriangleD), cudaMemcpyHostToDevice) );
        cudaSafe( cudaMemcpy(d_bvh_buffer, sceneData.h_models[i].bvh, sceneData.h_models[i].nrBvhNodes * sizeof(BVHNode), cudaMemcpyHostToDevice) );

        hd_models[i].trianglesV = d_vertex_buffer;
        hd_models[i].trianglesD = d_data_buffer;
        hd_models[i].bvh = d_bvh_buffer;
    }

    // Upload the collection of buffers to a new buffer
    Model* d_models;
    cudaSafe( cudaMalloc(&d_models, sceneData.num_models * sizeof(Model)) );
    cudaSafe( cudaMemcpy(d_models, hd_models, sceneData.num_models * sizeof(Model), cudaMemcpyHostToDevice) );

    cudaSafe( cudaMalloc(&d_instances, sceneData.num_objects * sizeof(Instance)) );
    h_instances = (Instance*)malloc(sceneData.num_objects * sizeof(Instance));

    cudaSafe( cudaMalloc(&d_topBvh, sceneData.num_top_bvh_nodes * sizeof(TopLevelBVH)) );

    Material* matBuf;
    cudaSafe( cudaMalloc(&matBuf, sceneData.num_materials * sizeof(Material)));
    cudaSafe( cudaMemcpy(matBuf, sceneData.h_material_buffer, sceneData.num_materials * sizeof(Material), cudaMemcpyHostToDevice) );

    // Assign to the global binding sites
    cudaSafe( cudaMemcpyToSymbol(DModels, &d_models, sizeof(d_models)) );
    cudaSafe( cudaMemcpyToSymbol(DInstances, &d_instances, sizeof(d_instances)) );
    cudaSafe( cudaMemcpyToSymbol(DMaterials, &matBuf, sizeof(matBuf)) );
    cudaSafe( cudaMemcpyToSymbol(DTopBVH, &d_topBvh, sizeof(d_topBvh)) );

    // queue of rays for wavefront tracing
    rayQueue = AtomicQueue<RayPacked>(NR_PIXELS);
    rayQueue.syncToDevice(DRayQueue);
    shadowRayQueue = AtomicQueue<RayPacked>(NR_PIXELS);
    shadowRayQueue.syncToDevice(DShadowRayQueue);
    rayQueueNew = AtomicQueue<RayPacked>(NR_PIXELS);
    rayQueueNew.syncToDevice(DRayQueueNew);

    // Allocate trace state SOA
    cudaSafe( cudaMalloc(&traceBufSOA.masks, NR_PIXELS * sizeof(float4)) );
    cudaSafe( cudaMalloc(&traceBufSOA.accucolors, NR_PIXELS * sizeof(float4)) );
    cudaSafe( cudaMalloc(&traceBufSOA.lights, NR_PIXELS * sizeof(float4)) );

    cudaSafe( cudaMalloc(&intersectionBuf, NR_PIXELS * sizeof(HitInfoPacked)) );


    // Enable NEE by default
    HNEE = true;

    // Update all the gameobjects for frame 1, afterwards we will do it while
    // the cpu is idle
    for(int i=0; i<sceneData.num_objects; i++)
    {
        h_instances[i] = ConvertToInstance(sceneData.h_object_buffer[i]);
    }
    cudaSafe( cudaMemcpy(d_instances, h_instances, sceneData.num_objects * sizeof(Instance), cudaMemcpyHostToDevice) );
    BuildTopLevelBVH(sceneData.h_top_bvh, h_instances, sceneData.h_models, sceneData.num_objects);
    cudaSafe( cudaMemcpy(d_topBvh, sceneData.h_top_bvh, sceneData.num_top_bvh_nodes * sizeof(TopLevelBVH), cudaMemcpyHostToDevice) );
}

void Pathtracer::Draw(const Camera& camera, float currentTime, bool shouldClear)
{

    // Map the screen texture resource.
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
    // These are used by simply kernels only, so we max
    // out the block size.
    int tx = 16;
    int ty = 16;
    dim3 dimBlock(WINDOW_WIDTH/tx+1, WINDOW_HEIGHT/ty+1);
    dim3 dimThreads(tx,ty);


    // clear the ray queue and update on gpu
    if (shouldClear)
    {
        kernel_clear_screen<<<dimBlock, dimThreads>>>(inputSurfObj);

        // Update the sphere area lights
        dSphereLightBuffer.update(sceneData.h_sphere_light_buffer);

        // sync NEE toggle
        cudaSafe( cudaMemcpyToSymbol(DNEE, &HNEE, sizeof(HNEE)) );
    }


    kernel_clear_state<<<NR_PIXELS/1024, 1024>>>(traceBufSOA);

    // Generate primary rays in the ray queue
    //rayQueue.clear();
    //rayQueue.syncToDevice(DRayQueue);
    //cudaSafe ( cudaDeviceSynchronize() );
    
    kernel_clear_rays<<<1,1>>>();
    kernel_generate_primary_rays<<<dimBlock, dimThreads>>>(camera, currentTime);


    uint max_bounces;
    if (_NEE)
        max_bounces = shouldClear ? 1 : 10;
    else
        max_bounces = shouldClear ? 2 : 10;

    for(int bounce = 0; bounce < max_bounces; bounce++) {

        // Test for intersections with each of the rays,
        kernel_extend<<<NR_PIXELS / 64 + 1, 64>>>(intersectionBuf, bounce);

        // Foreach intersection, possibly create shadow rays and secondary rays.
        kernel_shade<<<NR_PIXELS / 128 + 1, 128>>>(intersectionBuf, traceBufSOA, glfwGetTime(), bounce, dSkydomeTex);

        // Sample the light source for every shadow ray
        if (_NEE) kernel_connect<<<NR_PIXELS / 128 + 1, 128>>>(traceBufSOA);

        // swap the ray buffers and clear.
        kernel_swap_and_clear<<<1,1>>>();
    }

    // Write the final state accumulator into the texture
    kernel_add_to_screen<<<NR_PIXELS / 1024 + 1, 1024>>>(traceBufSOA, inputSurfObj);

    // Update all the gameobjects while the GPU is busy
    for(int i=0; i<sceneData.num_objects; i++)
    {
        h_instances[i] = ConvertToInstance(sceneData.h_object_buffer[i]);
    }
    BuildTopLevelBVH(sceneData.h_top_bvh, h_instances, sceneData.h_models, sceneData.num_objects);

    cudaSafe( cudaMemcpyAsync(d_instances, h_instances, sceneData.num_objects * sizeof(Instance), cudaMemcpyHostToDevice) );
    cudaSafe( cudaMemcpyAsync(d_topBvh, sceneData.h_top_bvh, sceneData.num_top_bvh_nodes * sizeof(TopLevelBVH), cudaMemcpyHostToDevice) );


    cudaSafe ( cudaDeviceSynchronize() );

    // Unmap the resource from cuda
    cudaSafe ( cudaGraphicsUnmapResources(1, &pGraphicsResource) );
}

#endif
