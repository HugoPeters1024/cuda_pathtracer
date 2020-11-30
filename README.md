## Cuda pathtracer
##### Hugo Peters (5927727)
##### November 2020

This the small report accompanying the 1st delivery for the Advanced Graphics course 2020-2021. For this project I have not worked with a partner.

### Dependencies

- nvidia-cuda-toolkit (apt) (tested on cuda 10 on sm_50 on a 960M, might have compatability issues on other platforms as I haven't tested these.
- cmake >3.16
- OpenMP
- GLFW3 (source included as build dependency, no action required)

### How to build

Inside the directory of the project you need to initialize the cmake cache the first time you build.  `cmake .`

Then you can build with `cmake --build .` which creates the `pathtracer` executable.

### Quick description of the controls

- WASD        - Move the camera
- Arrow Keys  - Change the direction of the camera
- QE          - Move the light source up and down
- UI          - Grow and shrink the area light
- N           - Toggle Next Event Estimation
- Space       - Toggle Raytracer /  Pathtracer

### Quicklist of the features

- Interactive Control
- Realtime pathtracer using cuda
- Glass, dielectrics and absorption
- obj loader and BVH builder using the SAH heuristic
- Efficient BVH traversal on the GPU.
- Skydome with artificial HDR capability
- Single configurable area light for the pathtracer
- Next Event Estimation for diffuse surfaces


### Implementation of the raytracer

The raytracer was implemented after the pathtracer. Although the pathtracer was implemented in cuda, 
I decided to implement the raytracer on the cpu. That way I could use recursion arbitrarily since 
it would have been very difficult to unroll the non tail recursive raytracer algorithm to run on 
graphics hardware efficiently. OpenMP was used as a simple and effective way to  parallelize the raytracer
dynamically on 8 threads. Given that the algorithm is embarrassingly parallel that result in a significant
speedup allowing an semi interactive experience.

Unlike the recursive parts of the whitted style radiance function many all intersection code and BVH traversal
from the cuda implementation could reasonably easy be adjusted to compile both a device and host function.
For these function many external sources where used in order to focus more on the radiance algorithm than the math.
So is the triangle intersection code adopted from scratchapixel[^ref1]

### Implementation of the pathtracer

The pathtracer was implemented with a bit more thought regarding performance. First of all and most obvious is
that fact that it runs on the GPU using Cuda 10. Moreover, I process the rays in a wavefront manner as described
by Jacco on his blog (which intern is based on a paper by NVIDIA[^ref2]). The main benefit is especially felt
in scenes where a significant number of paths end prematurely. I expect that this choice of architecture will
be even more beneficial when Russian Roulette is implemented in the future.

Making the trace algorithm iterative was done by leaning on the concept of a mask and accumulator color 
to substitute the recursion state as inspired by this blogpost by Sam Lepere[^ref3]. However, I found that I needed a
few more properties in the state of a pixel. The complete type looks like this:

```cpp
struct __align__(16) TraceState
{
    float3 mask;
    float3 accucolor;
    float3 currentNormal;
    // contains the inverse of the last applied lambert multiplication.
    float correction;
    bool fromSpecular;
};
```

The reason for this extra state requirement is due to my interpretation of wavefront pathtracing where the 'extend' kernel,
which processes the shadowrays, needs to know the surface normal, and how to undo the already applied lambert term. Furthermore,
because of Next Event Estimation is it necessary to know whether the previous bounce was a specular interaction, which is an
indication that all we can do is fall back to just one standard ray. In the future MIS might play a role here but more research
is needed.

### References

[^ref1]: [Triangle intersection](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection)
[^ref2]: [Wavefront pathtracing](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
[^ref3]: [Iterative tracing](http://raytracey.blogspot.com/2015/12/gpu-path-tracing-tutorial-2-interactive.html)






