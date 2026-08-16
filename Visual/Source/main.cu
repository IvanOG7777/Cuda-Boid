//
// Created by elder on 8/14/2026.
//

#include <iostream>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include <cuda_gl_interop.h>

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

    GLuint VAO = 0, VBO = 0;
    setVAO(VAO, VBO, GL_DYNAMIC_DRAW);

    const char* vertexShader = makeVertexShader();
    const char* fragmentShader = makeFragmentShader();

    GLuint VS = compileShader(vertexShader, GL_VERTEX_SHADER);
    GLuint FS = compileShader(fragmentShader, GL_FRAGMENT_SHADER);

    GLint program = glCreateProgram();
    glAttachShader(program, VS);
    glAttachShader(program, FS);
    glLinkProgram(program);
    glDeleteShader(VS);
    glDeleteShader(FS);

    Boid *deviceBoids = nullptr;
    float2 *deviceAccelerations = nullptr;
    curandState *deviceStates = nullptr;
    cudaError err = {};

    err = cudaMalloc(&deviceBoids, N_BOIDS * sizeof(Boid));
    if (err != cudaSuccess) {
        printf("Failed to allocated memory for device boids\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&deviceAccelerations, N_BOIDS * sizeof(float2));
    if (err != cudaSuccess) {
        printf("Failed to allocated memory for device accelerations\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&deviceStates, N_BOIDS * sizeof(curandState));
    if (err != cudaSuccess) {
        printf("Failed to allocated memory for device states\n");
        exit(EXIT_FAILURE);
    }

    cudaGraphicsResource *cudaResource;

    cudaGraphicsGLRegisterBuffer(&cudaResource, VBO, cudaGraphicsMapFlagsWriteDiscard);
    size_t bytes = 0;
    cudaGraphicsMapResources(1, &cudaResource, nullptr);
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&deviceBoids), &bytes, cudaResource);
    kernelInitBoids<<<BLOCKS, TPB>>>(deviceBoids, deviceStates, 123ULL);
    cudaDeviceSynchronize();
    cudaGraphicsMapResources(1, &cudaResource, nullptr);


    glEnable(GL_PROGRAM_POINT_SIZE);
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        size_t numBytes = 0;

        cudaGraphicsMapResources(1, &cudaResource, nullptr);

        cudaGraphicsResourceGetMappedPointer((void**)&cudaResource, &numBytes, cudaResource);

        kernelMakeAcceleration<<<BLOCKS, TPB>>>(deviceBoids, deviceAccelerations);
        cudaDeviceSynchronize();
        kernelIntegrateBoids<<<BLOCKS, TPB>>>(deviceBoids, deviceAccelerations);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &cudaResource, nullptr);

        glUseProgram(program);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N_BOIDS);
        glBindVertexArray(0);

        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    cudaGraphicsUnregisterResource(cudaResource);
    glfwDestroyWindow(window);
    glfwTerminate();

    cudaFree(deviceBoids);
    cudaFree(deviceAccelerations);
    cudaFree(deviceStates);

    return 0;
}
