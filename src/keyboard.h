#ifndef H_KEYBOARD
#define H_KEYBOARD

#include <vector>
#include "types.h"

enum ACTION {
    MOVE_RIGHT,
    MOVE_LEFT,
    MOVE_FORWARD,
    MOVE_BACKWARD,
    MOVE_UP,
    MOVE_DOWN,

    LOOK_UP,
    LOOK_DOWN,
    LOOK_LEFT,
    LOOK_RIGHT,

    SWITCH_MODE,
    SWITCH_NEE,
    SWITCH_CACHE,
    SWITCH_CONVERGE,
    SWITCH_BLUR,

    ATTACH_0,
    ATTACH_1,
    ATTACH_2,
    ATTACH_3,
    ATTACH_4,
    ATTACH_5,
    ATTACH_6,
    ATTACH_7,
    ATTACH_8,
    ATTACH_9,

    FOCUSS,
};

class Keyboard
{
private:
    GLFWwindow* window;
    std::vector<int> key_map, old_key_map;
    std::map<ACTION, int> action_map;
    int map_size;

    void generateActionMap();

public:
    Keyboard(GLFWwindow* window);
    // These methods suffer from this bug: https://github.com/glfw/glfw/issues/747
    // A fix is to be released in 3.3.0
    bool isPressed(ACTION a) const; // Pressed in this tick
    bool isReleased(ACTION a) const; // Released in this tick

    void swapBuffers();
    bool isDown(ACTION a) const;  // Down at all
};

Keyboard::Keyboard(GLFWwindow* window)
{
    this->window = window;

    // Preserves keypresses until polled to prevent missing events
    // This implicitly requires that all keys are polled to maintain
    // normal behavior
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

    map_size = GLFW_KEY_LAST + 1;
    key_map = std::vector<int>(map_size);
    old_key_map = std::vector<int>(map_size);
    action_map = std::map<ACTION, int>();
    generateActionMap();
}

void Keyboard::swapBuffers()
{
    // Keys below barcode 32 are not in use and generate errors.
    for(int i=32; i<map_size; i++)
    {
        old_key_map[i] = key_map[i];
        key_map[i] = glfwGetKey(window, i);
    }
}

bool Keyboard::isDown(ACTION a) const
{
    int key = action_map.at(a);
    return key_map[key] == GLFW_PRESS;
}

bool Keyboard::isPressed(ACTION a) const
{
    int key = action_map.at(a);
    return key_map[key] == GLFW_PRESS && old_key_map[key] != GLFW_PRESS;
}

bool Keyboard::isReleased(ACTION a) const
{
    int key = action_map.at(a);
    return key_map[key] != GLFW_PRESS && old_key_map[key] == GLFW_PRESS;
}


void Keyboard::generateActionMap()
{
    action_map[MOVE_LEFT]     = GLFW_KEY_A;
    action_map[MOVE_RIGHT]    = GLFW_KEY_D;
    action_map[MOVE_FORWARD]  = GLFW_KEY_W;
    action_map[MOVE_BACKWARD] = GLFW_KEY_S;
    action_map[MOVE_UP]       = GLFW_KEY_Q;
    action_map[MOVE_DOWN]     = GLFW_KEY_E;

    action_map[LOOK_UP]       = GLFW_KEY_UP;
    action_map[LOOK_DOWN]     = GLFW_KEY_DOWN;
    action_map[LOOK_LEFT]     = GLFW_KEY_LEFT;
    action_map[LOOK_RIGHT]    = GLFW_KEY_RIGHT;

    action_map[SWITCH_MODE]   = GLFW_KEY_SPACE;
    action_map[SWITCH_NEE]    = GLFW_KEY_N;
    action_map[SWITCH_CACHE]  = GLFW_KEY_C;
    action_map[SWITCH_CONVERGE]  = GLFW_KEY_CAPS_LOCK;
    action_map[SWITCH_BLUR]  = GLFW_KEY_B;

    action_map[ATTACH_0] = GLFW_KEY_0;
    action_map[ATTACH_1] = GLFW_KEY_1;
    action_map[ATTACH_2] = GLFW_KEY_2;
    action_map[ATTACH_3] = GLFW_KEY_3;
    action_map[ATTACH_4] = GLFW_KEY_4;
    action_map[ATTACH_5] = GLFW_KEY_5;
    action_map[ATTACH_6] = GLFW_KEY_6;
    action_map[ATTACH_7] = GLFW_KEY_7;
    action_map[ATTACH_8] = GLFW_KEY_8;
    action_map[ATTACH_9] = GLFW_KEY_9;

    action_map[FOCUSS] = GLFW_KEY_X;
}
#endif
