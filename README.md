## Cuda pathtracer
##### Hugo Peters (5927727)
##### December 2020

This the small report accompanying the 1st delivery for the Advanced Graphics course 2020-2021. For this project I have not worked with a partner.

## Dependencies

- nvidia-cuda-toolkit (apt) (tested on cuda 10 on sm_50 on a 960M, might have compatability issues on other platforms as I haven't tested these.)
- cmake >3.16
- OpenMP
- GLFW3 (source included as build dependency, no action required)

## How to build

Inside the directory of the project you need to initialize the cmake cache the first time you build.  `cmake .`

Then you can build with `cmake --build .` which creates the `pathtracer` executable.

## Quick description of the controls

- WASD        - Move the camera
- Arrow Keys  - Change the direction of the camera
- QE          - Move the light source up and down
- UI          - Grow and shrink the area light with index 0
- N           - Toggle Next Event Estimation
- Space       - Toggle Raytracer /  Pathtracer
- Page UP     - Widen aperture
- Page DOWN   - Close the aperture
- Left Mouse Click - Set the focal distance to the object you click on (aka focus on that object)

To select a scene pass a value to the `--scene` command line option. The following scenes are available:

- outside (default)
- sibenik 
- minecraft
- 2mtris
- example_scene.chai / `<path/to/script.chai>`

## new in version 3

- Bokeh by sampling the lens (offsetting the orgin and then refocussing on the focal plane)
- Using blue noise in the first 10 samples before continuing fully randomized. Especially effective on the rough edges of the normal maps.
- Further improved performance by reducing stack traffic and branching in the traversal algorithm.
- Actually correct results. Previous version had a temporal correlation in the seed.
- A CDF over the skydome has been implemented but disabled for performance reasons.
- An attempt to sort the rays in eight buckets based the octant of their direction vector but this did not yield a netto performance increase. Quite a significant slowdown in fact.

## new in version 2

** VIDEOS IN SCREENSHOT DIRECTORY **

- Skydome is now hdr
- BVH with binning approach (<700ms for 2M triangles single core)
- SIMD instructions here and there
- No more round trips to the GPU during rendering
- Render performance x3 (130 fps 1 diffuse, 1 shadow ray in sibenik)
- Toplevel BVH with full transformation support
- Russian Roulette (now 30+ fps with 1spp in sibenik)


---------------------------------------------------------
<img src="./screenshots/2mtris.png" width="40%" />
<img src="./screenshots/2mtris_render.png" width="40%" />

Building a BVH over 2M triangles under 1 second

-----------------------------------------------------------------

For a demo of the toplevel bvh please check out the video's in the screenshot directory.
Here  is an impression of the code structure:

<img src="./screenshots/callbacks.png" width="100%" />

------------------------------

<img src="./screenshots/minecraft.png" width="40%" />

Textures with alpha channels are a WIP

-------------------------------

<img src="./screenshots/textures.png" width="40%" />
<img src="./screenshots/windows.png" width="40%" />

The sibenik cathedral in it's full glory with textures and normal maps.


## Quicklist of the features

- Interactive raytracer using OpenMP
- Realtime pathtracer using cuda
- Multiple scenes
- Glass, dielectrics and absorption
- obj loader and BVH builder using the SAH heuristic
- Efficient BVH traversal on the GPU.
- Multiple configurable point lights for the raytracer
- Multiple configurable area lights for the pathtracer
- Anti Aliasing for the pathtracer (for raytracer I valued interactivity more)
- Next Event Estimation for diffuse surfaces
- Skydome
- Post processing shader with gamma correction, chromatic aberration, and vignetting
- Barrel distortion at the primary ray level [^ref4]



## Implementation of the raytracer

The raytracer was implemented after the pathtracer. Although the pathtracer was implemented in cuda, 
I decided to implement the raytracer on the cpu. That way I could use recursion arbitrarily since 
it would have been very difficult to unroll the non tail recursive raytracer algorithm to run on 
graphics hardware efficiently. OpenMP was used as a simple and effective way to  parallelize the raytracer
dynamically on 8 threads. Given that the algorithm is embarrassingly parallel that resulted in a significant
speedup allowing a semi interactive experience.

Unlike the recursive parts of the whitted style radiance function, almost all intersection- and BVH traversal code
from the cuda implementation could reasonably easy be adjusted to compile both a device and host function.
For these function many external sources where used in order to focus more on the radiance algorithm than the math.
So is the triangle intersection code adopted from scratchapixel[^ref1]

## Implementation of the pathtracer

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

### Screenshots

<img src="./screenshots/sibenik1.png" width="40%" />
<img src="./screenshots/sibenik1_NEE.png" width="40%" />

Comparision of Next Even Estimation off or on (left to right), both images are taken after about 60 seconds rendering on the sibenik scene. 
But since the version without NEE runs about 1.5 faster (because of the lack of shadowrays) the image on the left actually has about 
1.5 times more samples.

---

<img src="./screenshots/outside1_ray.png" width="40%" />
<img src="./screenshots/outside1_path.png" width="40%" />

Comparasion of raytracing (left) vs pathtracer (right). Note that the skybox is currently only supported in pathtraced mode. 
But a constant colored skydome was selected for the raytracer to emulate similar lighting conditions.

---

<img src="./screenshots/sibenik_final.png" width="80%" />

Finally that sibenik scene again with all post processing effects enabled. As well as a tilted glass cube with absorption for style.
Cheers Beer!

### Wishlist for the future

- Barrel distortion
- MIS 
- Spotlights
- Faster, Faster
- Loading screen
- Depth of field

### Closing remarks

The result is already much better than I had hoped for. It has cost many hours of labour but it was so much fun that I had no
problem pouring my soul into this project. I am also very excited to see what the rest of the course brings and how I can incorporate
more neat little tricks to make this project even better.

### References

[^ref1]: [Triangle intersection](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection)
[^ref2]: [Wavefront pathtracing](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
[^ref3]: [Iterative tracing](http://raytracey.blogspot.com/2015/12/gpu-path-tracing-tutorial-2-interactive.html)
[^ref4]: [Barrel distortion](https://www.imatest.com/docs/distortion_instructions/)
[^ref5]: [Optix Slab Test](http://jcgt.org/published/0007/03/04/)

