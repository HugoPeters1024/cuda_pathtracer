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
    HitInfoPacked* intersectionBuf;
    TraceStateSOA traceBufSOA;
    CudaTexture dSkydomeTex;
    CDF d_skydomeCDF;
    Instance* d_instances;
    TopLevelBVH* d_topBvh;
    DSizedBuffer<TriangleLight> d_lights;


public:
    Pathtracer(Scene& scene, GLuint texture) : Application(scene, texture) {}

    virtual void Init() override;
    virtual void Render(const Camera& camera, float currentTime, float frameTime, bool shouldClear) override;
    virtual void Finish() override;
};

void Pathtracer::Init()
{
    float4* h_skydome;
    dSkydomeTex.texture_id = loadTextureHDR("cave.hdr", h_skydome, dSkydomeTex.width, dSkydomeTex.height);

    // Calculate the CDF
    CDF h_skydomeCDF;
    h_skydomeCDF.values = (float*)malloc(dSkydomeTex.width * dSkydomeTex.height * sizeof(float));
    h_skydomeCDF.cumValues = (float*)malloc(dSkydomeTex.width * dSkydomeTex.height * sizeof(float));
    h_skydomeCDF.nrItems = dSkydomeTex.width * dSkydomeTex.height;
    float totalEnergy = 0.0f;
    for(uint y=0; y<dSkydomeTex.height; y++)
    {
        for(uint x=0; x<dSkydomeTex.width; x++)
        {
            const float energy = fmaxcompf(get3f(h_skydome[x + dSkydomeTex.width * y]));
            h_skydomeCDF.values[x + dSkydomeTex.width * y] = energy;
            totalEnergy += energy;
            h_skydomeCDF.cumValues[x + dSkydomeTex.width * y] = totalEnergy;
        }
    }

    for(uint y=0; y<dSkydomeTex.height; y++)
    {
        for(uint x=0; x<dSkydomeTex.width; x++)
        {
            h_skydomeCDF.values[x + dSkydomeTex.width * y] /= totalEnergy;
            h_skydomeCDF.cumValues[x + dSkydomeTex.width * y] /= totalEnergy;
        }
    }

    printf("Total energy in skydome CDF: %f\n", totalEnergy);

    free(h_skydome);

    d_skydomeCDF.nrItems = h_skydomeCDF.nrItems;
    cudaSafe( cudaMalloc(&d_skydomeCDF.values, dSkydomeTex.width * dSkydomeTex.height * sizeof(float)) );
    cudaSafe( cudaMalloc(&d_skydomeCDF.cumValues, dSkydomeTex.width * dSkydomeTex.height * sizeof(float)) );
    cudaSafe( cudaMemcpy(d_skydomeCDF.values, h_skydomeCDF.values, dSkydomeTex.width * dSkydomeTex.height * sizeof(float), cudaMemcpyHostToDevice) );
    cudaSafe( cudaMemcpy(d_skydomeCDF.cumValues, h_skydomeCDF.cumValues, dSkydomeTex.width * dSkydomeTex.height * sizeof(float), cudaMemcpyHostToDevice) );

    free(h_skydomeCDF.values);
    free(h_skydomeCDF.cumValues);


    // Register the texture with cuda as preperation for interop.
    cudaSafe( cudaGraphicsGLRegisterImage(&pGraphicsResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore) );

    dSphereBuffer = DSizedBuffer<Sphere>(scene.spheres.data(), scene.spheres.size(), &DSpheres);
    dPlaneBuffer = DSizedBuffer<Plane>(scene.planes.data(), scene.planes.size(), &DPlanes);

    // Host buffer of device buffers... ;)
    Model hd_models[scene.models.size()];

    // copy over the model but set the gpu buffers as source
    memcpy(hd_models, scene.models.data(), scene.models.size() * sizeof(Model));

    for(int i=0; i<scene.models.size(); i++)
    {
        BVHNode* d_bvh_buffer;

        // Upload the host buffers to cuda
        cudaSafe( cudaMalloc(&d_bvh_buffer, scene.models[i].nrBvhNodes * sizeof(TriangleD)) );
        cudaSafe( cudaMemcpy(d_bvh_buffer, scene.models[i].bvh, scene.models[i].nrBvhNodes * sizeof(BVHNode), cudaMemcpyHostToDevice) );

        hd_models[i].bvh = d_bvh_buffer;
    }

    // Extract all emissive triangles for explicit sampling
    std::vector<TriangleLight> lights;
    for(uint i=0; i<scene.objects.size(); i++)
    {
        const Model& model = scene.models[scene.instances[i].model_id];
        for(uint t=model.triangleStart; t<model.triangleStart+model.nrTriangles; t++)
        {
            const Material& mat = scene.materials[scene.allVertexData[t].material];
            if (fmaxcompf(mat.emission) < EPS) continue;
            lights.push_back(TriangleLight { t, i });
        }
    }

    DSizedBuffer<TriangleLight>(lights.data(), lights.size(), &DTriangleLights);
    printf("Extracted %lu emmissive triangles from the scene\n", lights.size());

    // Upload the collection of buffers to a new buffer
    TriangleV* d_vertices;
    cudaSafe( cudaMalloc(&d_vertices, scene.allVertices.size() * sizeof(TriangleV)) );
    cudaSafe( cudaMemcpy(d_vertices, scene.allVertices.data(), scene.allVertices.size() * sizeof(TriangleV), cudaMemcpyHostToDevice) );

    TriangleV* d_vertexData;
    cudaSafe( cudaMalloc(&d_vertexData, scene.allVertexData.size() * sizeof(TriangleD)) );
    cudaSafe( cudaMemcpy(d_vertexData, scene.allVertexData.data(), scene.allVertexData.size() * sizeof(TriangleD), cudaMemcpyHostToDevice) );

    Model* d_models;
    cudaSafe( cudaMalloc(&d_models, scene.models.size() * sizeof(Model)) );
    cudaSafe( cudaMemcpy(d_models, hd_models, scene.models.size() * sizeof(Model), cudaMemcpyHostToDevice) );

    cudaSafe( cudaMalloc(&d_instances, scene.objects.size() * sizeof(Instance)) );

    cudaSafe( cudaMalloc(&d_topBvh, scene.topLevelBVH.size() * sizeof(TopLevelBVH)) );

    Material* matBuf;
    cudaSafe( cudaMalloc(&matBuf, scene.materials.size() * sizeof(Material)));
    cudaSafe( cudaMemcpy(matBuf, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice) );

    // Assign to the global binding sites
    cudaSafe( cudaMemcpyToSymbol(DVertices, &d_vertices, sizeof(d_models)) );
    cudaSafe( cudaMemcpyToSymbol(DVertexData, &d_vertexData, sizeof(d_models)) );
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

    cudaSafe( cudaMemcpy(d_instances, scene.instances, scene.objects.size() * sizeof(Instance), cudaMemcpyHostToDevice) );
    cudaSafe( cudaMemcpy(d_topBvh, scene.topLevelBVH.data(), scene.topLevelBVH.size() * sizeof(TopLevelBVH), cudaMemcpyHostToDevice) );
}

void Pathtracer::Render(const Camera& camera, float currentTime, float frameTime, bool shouldClear)
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
        // sync NEE toggle
        cudaSafe( cudaMemcpyToSymbolAsync(DNEE, &HNEE, sizeof(HNEE)) );

        cudaSafe( cudaMemcpyAsync(d_instances, scene.instances, scene.objects.size() * sizeof(Instance), cudaMemcpyHostToDevice) );
        cudaSafe( cudaMemcpyAsync(d_topBvh, scene.topLevelBVH.data(), scene.topLevelBVH.size() * sizeof(TopLevelBVH), cudaMemcpyHostToDevice) );

        kernel_clear_screen<<<dimBlock, dimThreads>>>(inputSurfObj);
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
        max_bounces = shouldClear ? 1 : 64;
    else
        max_bounces = shouldClear ? 2 : 64;

    for(int bounce = 0; bounce < max_bounces; bounce++) {

        // Test for intersections with each of the rays,
        uint kz = 64;
        kernel_extend<<<NR_PIXELS / kz + 1, kz>>>(intersectionBuf, bounce);

        // Foreach intersection, possibly create shadow rays and secondary rays.
        kernel_shade<<<NR_PIXELS / 64 + 1, 64>>>(intersectionBuf, traceBufSOA, glfwGetTime(), bounce, dSkydomeTex, d_skydomeCDF);

        // Sample the light source for every shadow ray
        if (_NEE) kernel_connect<<<NR_PIXELS / kz + 1, kz>>>(traceBufSOA);

        // swap the ray buffers and clear.
        kernel_swap_and_clear<<<1,1>>>();
    }

    // Write the final state accumulator into the texture
    kernel_add_to_screen<<<NR_PIXELS / 1024 + 1, 1024>>>(traceBufSOA, inputSurfObj);
}


void Pathtracer::Finish()
{
    cudaSafe ( cudaDeviceSynchronize() );

    // Unmap the resource from cuda
    cudaSafe ( cudaGraphicsUnmapResources(1, &pGraphicsResource) );
}

#endif
