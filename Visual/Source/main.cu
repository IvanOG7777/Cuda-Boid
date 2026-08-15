//
// Created by elder on 8/14/2026.
//

#include <curand_kernel.h>
#include <iostream>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "../Header/deviceFunctions.cuh"
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

    GLuint VBO = 0, VAO = 0;

    Boid *deviceBoids = nullptr;
    curandState *deviceStates = nullptr;
    cudaError err = {};

    err = cudaMalloc(&deviceBoids, N_BOIDS * sizeof(Boid));
    if (err != cudaSuccess) {
        printf("Failed to allocated memory for device boids\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&deviceStates, N_BOIDS * sizeof(curandState));
    if (err != cudaSuccess) {
        printf("Failed to allocated memory for device states\n");
        exit(EXIT_FAILURE);
    }

    kernelInitBoids<<<BLOCKS, TPB>>>(deviceBoids, deviceStates, 123ULL);
    cudaDeviceSynchronize();


    return 0;
}
