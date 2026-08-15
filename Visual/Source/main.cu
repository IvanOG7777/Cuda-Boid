//
// Created by elder on 8/14/2026.
//

#include <iostream>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "../Header/glUtils.cuh"

int main() {

    if (!glfwInit()) {
        std:: cerr << "FAILED TO LOAD GLFW\n";
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = createWindow(1280, 720, "Boid Visual");
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "GLAD INIT ERROR\n";
        return -1;
    }

    return 0;
}
