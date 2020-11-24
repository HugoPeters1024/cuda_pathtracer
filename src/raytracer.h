#ifndef H_RAYTRACER
#define H_RAYTRACER

#include "types.h"
#include "constants.h"
#include "application.h"
#include "globals.h"
#include "kernels.h"

class Raytracer : public Application
{
private:
    float* screenBuffer;

public:
    Raytracer(SceneData sceneData, GLuint texture) : Application(sceneData, texture) {}
    virtual void Init() override;
    virtual void Draw(const Camera& camera, float currentTime, bool shouldClear) override;
};

void Raytracer::Init()
{
    screenBuffer = (float*)malloc(4 * NR_PIXELS * sizeof(float));
}

void Raytracer::Draw(const Camera& camera, float currentTime, bool shouldClear)
{
    for(uint y=0; y<WINDOW_HEIGHT; y++)
    {
        for(uint x=0; x<WINDOW_WIDTH*4; x+=4)
        {
            screenBuffer[x + y * 4 * WINDOW_WIDTH + 0] = currentTime / 10.0f;
            screenBuffer[x + y * 4 * WINDOW_WIDTH + 1] = currentTime / 10.0f;
            screenBuffer[x + y * 4 * WINDOW_WIDTH + 2] = currentTime / 10.0f;
            screenBuffer[x + y * 4 * WINDOW_WIDTH + 3] = 1;
        }
    }
    glBindTexture(GL_TEXTURE_2D, texture);
    glTextureSubImage2D(texture, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_FLOAT, screenBuffer);
    glBindTexture(GL_TEXTURE_2D, 0);
}

#endif
