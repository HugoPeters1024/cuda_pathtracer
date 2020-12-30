#ifndef H_STATE_LOADER
#define H_STATE_LOADER

#include <iostream>
#include <sstream>

#include "types.h"

static float3 parseFloat3(const std::string& line)
{
    float3 ret;
    char tmp;
    std::istringstream stream(line);
    stream >> ret.x;
    stream >> tmp;
    stream >> ret.y;
    stream >> tmp;
    stream >> ret.z;
    return ret;
}

static float parseFloat(const std::string& line)
{
    float ret;
    std::istringstream stream(line);
    stream >> ret;
    return ret;
}

static Camera initialCamera() 
{
    return Camera(make_float3(0,2,-3), make_float3(0,0,1), 1.5);
}

static void saveState(const Camera& camera)
{
    std::ofstream file;
    file.open("save.txt");
    if (file.is_open())
    {
        file << camera.eye.x << "|" << camera.eye.y << "|" << camera.eye.z << std::endl;
        file << camera.viewDir.x << "|" << camera.viewDir.y << "|" << camera.viewDir.z << std::endl;
        file << camera.d;
        file.close();
    }
    else fprintf(stderr, "unable to open state file while saving\n");
}

static Camera readState()
{
    std::ifstream file;
    file.open("save.txt");
    std::string line;
    char tmp;
    if (file.is_open())
    {
        getline(file, line);
        float3 eye = parseFloat3(line);
        getline(file, line);
        float3 viewDir = parseFloat3(line);
        getline(file, line);
        float d = parseFloat(line);
        return Camera(eye, viewDir, d);
    } 
    else 
    {
        fprintf(stderr, "unable to open state file for reading, creating new one\n");
        return initialCamera();
    }
}

#endif
