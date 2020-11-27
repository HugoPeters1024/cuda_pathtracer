#ifndef SHADERS_H
#define SHADERS_H

#include "types.h"


#include <array>
#include <string>
#include <sstream>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <regex>

inline static GLuint CompileShader(GLint type, const char* source)
{
  // Preprocess macro's
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);

  // Check the compilation of the shader 
  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

  if (success == GL_FALSE)
  {
    GLint maxLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

    GLchar* errorLog = (GLchar*)malloc(maxLength);
    glGetShaderInfoLog(shader, maxLength, &maxLength, errorLog);

    printf("Error in shader: %s", errorLog);

    free(errorLog);
    return -1;
  }

  return shader;
};

inline static bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

inline static GLuint CompileShaderFile(GLint type, const char* filename)
{
    std::ifstream t(filename);
    std::string src((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    std::ifstream structures_t("shaders/structures.glsl");
    std::string structures((std::istreambuf_iterator<char>(structures_t)), std::istreambuf_iterator<char>());

    std::ifstream random_t("shaders/random.glsl");
    std::string random((std::istreambuf_iterator<char>(random_t)), std::istreambuf_iterator<char>());

    replace(src, "#STRUCTURES", structures);
    replace(src, "#RANDOM", random);
    GLuint ret = CompileShader(type, src.c_str());
    if (ret == -1)
        printf("^ for file %s\n", filename);
    return ret;
}

inline static GLuint GenerateProgram(GLuint cs)
{
  GLuint program = glCreateProgram();
  glAttachShader(program, cs);
  glLinkProgram(program);
  GLint isLinked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
  if (!isLinked)
  {
    GLint maxLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
    GLchar* errorLog = (GLchar*)malloc(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, errorLog);

    printf("Shader linker error: %s", errorLog);

    glDeleteProgram(program);
    exit(5);
  }
  return program;
}

inline static GLuint GenerateProgram(GLuint vs, GLuint fs)
{
  GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  GLint isLinked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
  if (!isLinked)
  {
    GLint maxLength = 0;  
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
    GLchar* errorLog = (GLchar*)malloc(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, errorLog);

    printf("Shader linker error: %s", errorLog);

    glDeleteProgram(program);
    exit(5);
  }
  return program;
}

inline static GLuint GenerateProgram(GLuint vs, GLuint gs, GLuint fs)
{
  GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, gs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  GLint isLinked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
  if (!isLinked)
  {
    GLint maxLength = 0;  
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
    GLchar* errorLog = (GLchar*)malloc(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, errorLog);

    printf("Shader linker error: %s", errorLog);

    glDeleteProgram(program);
    exit(5);
  }
  return program;
}

#endif
