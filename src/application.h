#ifndef H_APPLICATION
#define H_APPLICATION

#include "types.h"
#include "scene.h"

class Application
{
protected:
    Scene& scene;
    GLuint texture;

public:
    Application(Scene& scene, GLuint texture) : scene(scene), texture(texture) {}
    virtual void Init() = 0;
    virtual void Render(const Camera& camera, float currentTime, float frameTime, bool shouldClear, uint sample) = 0;
    virtual void Finish() = 0;
};


#endif
