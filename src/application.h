#ifndef H_APPLICATION
#define H_APPLICATION

#include "types.h"
#include "scene.h"

class Application
{
protected:
    Scene& scene;
    GLuint luminanceTexture;
    GLuint albedoTexture;

public:
    Application(Scene& scene, GLuint luminanceTexture, GLuint albedoTexture) 
        : scene(scene), luminanceTexture(luminanceTexture), albedoTexture(albedoTexture)  {}
    virtual void Init() = 0;
    virtual void Render(const Camera& camera, float currentTime, float frameTime, bool shouldClear) = 0;
    virtual void Finish() = 0;
};


#endif
