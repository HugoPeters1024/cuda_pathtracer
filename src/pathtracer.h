#ifndef H_PATHTRACER
#define H_PATHTRACER

#include "application.h"
#include "globals.h"
#include "kernels.h"

class Pathtracer : public Application
{
private:
    cudaGraphicsResource* pGraphicsResource;
    cudaArray *arrayPtr;
    AtomicQueue<Ray> rayQueue;
    AtomicQueue<Ray> shadowRayQueue;
    AtomicQueue<Ray> rayQueueNew;
    HitInfo* intersectionBuf;
    TraceState* traceBuf;

public:
    Pathtracer(SceneData sceneData, GLuint texture) : Application(sceneData, texture) {}

    virtual void Init() override;
    virtual void Draw(const Camera& camera, float currentTime, bool shouldClear) override;
};

void Pathtracer::Init()
{
    // Register the texture with cuda as preperation for interop.
    cudaSafe( cudaGraphicsGLRegisterImage(&pGraphicsResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );

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

    Material* matBuf;
    cudaSafe( cudaMalloc(&matBuf, sceneData.num_materials * sizeof(Material)));
    cudaSafe( cudaMemcpy(matBuf, sceneData.h_material_buffer, sceneData.num_materials * sizeof(Material), cudaMemcpyHostToDevice) );

    // Assign to the global binding sites
    cudaSafe( cudaMemcpyToSymbol(DTriangles, &d_vertex_buffer, sizeof(d_vertex_buffer)) );
    cudaSafe( cudaMemcpyToSymbol(DTriangleData, &d_data_buffer, sizeof(d_data_buffer)) );
    cudaSafe( cudaMemcpyToSymbol(DBVH, &d_bvh_buffer, sizeof(d_bvh_buffer)) );
    cudaSafe( cudaMemcpyToSymbol(DMaterials, &matBuf, sizeof(matBuf)) );

    // queue of rays for wavefront tracing
    rayQueue = AtomicQueue<Ray>(NR_PIXELS);
    shadowRayQueue = AtomicQueue<Ray>(NR_PIXELS);
    rayQueueNew = AtomicQueue<Ray>(NR_PIXELS);

    cudaSafe( cudaMalloc(&intersectionBuf, NR_PIXELS * sizeof(HitInfo)) );
    cudaSafe( cudaMalloc(&traceBuf, NR_PIXELS * sizeof(TraceState)) );
}

void Pathtracer::Draw(const Camera& camera, float currentTime, bool shouldClear)
{
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


    // clear the ray queue and update on gpu
    if (shouldClear)
        kernel_clear_screen<<<dimBlock, dimThreads>>>(inputSurfObj);


    kernel_clear_state<<<NR_PIXELS/1024, 1024>>>(traceBuf);

    // Generate primary rays in the ray queue
    rayQueue.clear();
    rayQueue.syncToDevice(DRayQueue);
    kernel_generate_primary_rays<<<dimBlock, dimThreads>>>(camera, currentTime);
    rayQueue.syncFromDevice(DRayQueue);
    assert (rayQueue.size == WINDOW_WIDTH * WINDOW_HEIGHT);


    uint max_bounces = shouldClear ? 1 : 8;
    for(int bounce = 0; bounce < max_bounces; bounce++) {

        // Test for intersections with each of the rays,
        kernel_extend<<<rayQueue.size / 64 + 1, 64>>>(intersectionBuf);

        // Foreach intersection, possibly create shadow rays and secondary rays.
        shadowRayQueue.clear();
        shadowRayQueue.syncToDevice(DShadowRayQueue);
        rayQueueNew.clear();
        rayQueueNew.syncToDevice(DRayQueueNew);
        kernel_shade<<<rayQueue.size / 128 + 1, 128>>>(intersectionBuf, traceBuf, glfwGetTime(), bounce);
        shadowRayQueue.syncFromDevice(DShadowRayQueue);
        rayQueueNew.syncFromDevice(DRayQueueNew);

        // Sample the light source for every shadow ray
        kernel_connect<<<shadowRayQueue.size / 128 + 1, 128>>>(traceBuf);

        // swap the ray buffers
        rayQueueNew.syncToDevice(DRayQueue);
        std::swap(rayQueue, rayQueueNew);
    }

    // Write the final state accumulator into the texture
    kernel_add_to_screen<<<NR_PIXELS / 1024 + 1, 1024>>>(traceBuf, inputSurfObj);


    cudaSafe ( cudaDeviceSynchronize() );

    // Unmap the resource from cuda
    cudaSafe ( cudaGraphicsUnmapResources(1, &pGraphicsResource) );
}

#endif
