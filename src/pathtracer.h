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
    HitInfoPacked* intersectionBuf;
    TraceStateSOA traceBufSOA;
    CudaTexture dSkydomeTex;
    CDF d_skydomeCDF;
    DSizedBuffer<TriangleLight> d_lights;
    SampleCache* d_sampleCache;


public:
    SceneBuffers sceneBuffers;
    RandState randState;
    Pathtracer(Scene& scene, GLuint texture) : Application(scene, texture) {}

    virtual void Init() override;
    virtual void Render(const Camera& camera, float currentTime, float frameTime, bool shouldClear) override;
    virtual void Finish() override;
};

void Pathtracer::Init()
{
    randState.randIdx = 0;
    randState.sampleIdx = 0;
    int blueNoiseW, blueNoiseH;
    randState.blueNoise = loadTextureL("bluenoise.png", blueNoiseW, blueNoiseH);
    randState.blueNoiseSize = make_float2((float)blueNoiseW, (float)blueNoiseH);
    randState.blueNoiseOffset = make_float2(0);


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

    sceneBuffers.num_spheres = scene.spheres.size();
    cudaSafe( cudaMalloc(&sceneBuffers.spheres, sceneBuffers.num_spheres * sizeof(Sphere)) );
    cudaSafe( cudaMemcpy(sceneBuffers.spheres, scene.spheres.data(), sceneBuffers.num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice) );

    sceneBuffers.num_planes = scene.planes.size();
    cudaSafe( cudaMalloc(&sceneBuffers.planes, sceneBuffers.num_planes * sizeof(Plane)) );
    cudaSafe( cudaMemcpy(sceneBuffers.planes, scene.planes.data(), sceneBuffers.num_planes * sizeof(Plane), cudaMemcpyHostToDevice) );

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
        const Instance& instance = scene.instances[i];
        const Model& model = scene.models[instance.model_id];
        for(uint t=model.triangleStart; t<model.triangleStart+model.nrTriangles; t++)
        {
            const uint mat_id = instance.material_id == 0xffffffff ? scene.allVertexData[t].material : instance.material_id;
            const Material& mat = scene.materials[mat_id];
            if (fmaxcompf(mat.emission) < EPS) continue;
            lights.push_back(TriangleLight { t, i });
        }
    }

    DSizedBuffer<TriangleLight>(lights.data(), lights.size(), &DTriangleLights);
    printf("Extracted %lu emmissive triangles from the scene\n", lights.size());

    // Upload the collection of buffers to a new buffer
    sceneBuffers.num_triangles = scene.allVertices.size();
    cudaSafe( cudaMalloc(&sceneBuffers.vertices, scene.allVertices.size() * sizeof(TriangleV)) );
    cudaSafe( cudaMemcpy(sceneBuffers.vertices, scene.allVertices.data(), scene.allVertices.size() * sizeof(TriangleV), cudaMemcpyHostToDevice) );

    cudaSafe( cudaMalloc(&sceneBuffers.vertexData, scene.allVertexData.size() * sizeof(TriangleD)) );
    cudaSafe( cudaMemcpy(sceneBuffers.vertexData, scene.allVertexData.data(), scene.allVertexData.size() * sizeof(TriangleD), cudaMemcpyHostToDevice) );

    cudaSafe( cudaMalloc(&sceneBuffers.radianceCaches, scene.allVertices.size() * sizeof(RadianceCache)) );

    cudaSafe( cudaMalloc(&sceneBuffers.models, scene.models.size() * sizeof(Model)) );
    cudaSafe( cudaMemcpy(sceneBuffers.models, hd_models, scene.models.size() * sizeof(Model), cudaMemcpyHostToDevice) );

    cudaSafe( cudaMalloc(&sceneBuffers.instances, scene.objects.size() * sizeof(Instance)) );

    cudaSafe( cudaMalloc(&sceneBuffers.topBvh, scene.topLevelBVH.size() * sizeof(TopLevelBVH)) );

    cudaSafe( cudaMalloc(&sceneBuffers.materials, scene.materials.size() * sizeof(Material)));
    cudaSafe( cudaMemcpy(sceneBuffers.materials, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice) );

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

    // Allocate the buffer needed to update the cache buckets
    // 64 is the maximum raydepth;
    cudaSafe( cudaMalloc(&d_sampleCache, NR_PIXELS * sizeof(SampleCache) * MAX_CACHE_DEPTH) );


    // Enable NEE and Caching by default
    HNEE = true;
    HCACHE = true;

    cudaSafe( cudaMemcpy(sceneBuffers.instances, scene.instances, scene.objects.size() * sizeof(Instance), cudaMemcpyHostToDevice) );
    cudaSafe( cudaMemcpy(sceneBuffers.topBvh, scene.topLevelBVH.data(), scene.topLevelBVH.size() * sizeof(TopLevelBVH), cudaMemcpyHostToDevice) );

    printf("Initializing radiance caches...\n");
    kernel_init_radiance_cache<<<sceneBuffers.num_triangles/1024, 1024>>>(sceneBuffers);
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
        // sync toggles
        cudaSafe( cudaMemcpyToSymbolAsync(DNEE, &HNEE, sizeof(HNEE)) );
        cudaSafe( cudaMemcpyToSymbolAsync(DCACHE, &HCACHE, sizeof(HNEE)) );

        cudaSafe( cudaMemcpyAsync(sceneBuffers.instances, scene.instances, scene.objects.size() * sizeof(Instance), cudaMemcpyHostToDevice) );
        cudaSafe( cudaMemcpyAsync(sceneBuffers.topBvh, scene.topLevelBVH.data(), scene.topLevelBVH.size() * sizeof(TopLevelBVH), cudaMemcpyHostToDevice) );

        kernel_clear_screen<<<dimBlock, dimThreads>>>(inputSurfObj);
        randState.sampleIdx = 0;
    }


    for(uint sample = 0; sample < (shouldClear ? scene.interactive_depth : 1); sample++)
    {

        kernel_clear_state<<<NR_PIXELS/1024+1, 1024>>>(traceBufSOA);

        // Generate primary rays in the ray queue
        //rayQueue.clear();
        //rayQueue.syncToDevice(DRayQueue);
        //cudaSafe ( cudaDeviceSynchronize() );
        
        kernel_clear_rays<<<1,1>>>();
        kernel_generate_primary_rays<<<dimBlock, dimThreads>>>(camera, randState);
        randState.randIdx++;


        uint max_bounces;
        if (_NEE)
            max_bounces = shouldClear ? scene.interactive_depth : MAX_RAY_DEPTH;
        else
            max_bounces = shouldClear ? scene.interactive_depth+1 : MAX_RAY_DEPTH;

        for(int bounce = 0; bounce < max_bounces; bounce++) {

            // Test for intersections with each of the rays,
            uint kz = 64;
            kernel_extend<<<NR_PIXELS / kz + 1, kz>>>(sceneBuffers, intersectionBuf, bounce);

            // Foreach intersection, possibly create shadow rays and secondary rays.
            kernel_shade<<<NR_PIXELS / 64 + 1, 64>>>(sceneBuffers, intersectionBuf, traceBufSOA, randState, bounce, dSkydomeTex, d_skydomeCDF, d_sampleCache);
            randState.randIdx++;

            // Sample the light source for every shadow ray
            if (_NEE) kernel_connect<<<NR_PIXELS / kz + 1, kz>>>(sceneBuffers, traceBufSOA);

            // swap the ray buffers and clear.
            kernel_swap_and_clear<<<1,1>>>();
        }

        if (!shouldClear && HCACHE)
        {
            kernel_update_buckets<<<NR_PIXELS/512 + 1, 512>>>(sceneBuffers, traceBufSOA, d_sampleCache);
            kernel_propagate_buckets<<<sceneBuffers.num_triangles/512+1,512>>>(sceneBuffers);
        }

        // Write the final state accumulator into the texture
        kernel_add_to_screen<<<NR_PIXELS/1024 + 1, 1024>>>(traceBufSOA, inputSurfObj, randState);
        randState.sampleIdx++;
    }
}


void Pathtracer::Finish()
{
    cudaSafe ( cudaDeviceSynchronize() );

    // Unmap the resource from cuda
    cudaSafe ( cudaGraphicsUnmapResources(1, &pGraphicsResource) );
}

#endif
