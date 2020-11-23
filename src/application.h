#ifndef H_APPLICATION
#define H_APPLICATION

#include "types.h"
#include "scene.h"

class Application
{
protected:
    SceneData sceneData;
    GLuint texture;

public:
    Application(SceneData sceneData, GLuint texture) : sceneData(sceneData), texture(texture) {}
    virtual void Init() = 0;
    virtual void Draw(const Camera& camera, float currentTime, bool shouldClera) = 0;
};


#endif
