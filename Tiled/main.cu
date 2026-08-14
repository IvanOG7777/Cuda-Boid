//
// Created by elder on 8/12/2026.
//

#include <iostream>
#include "deviceFunctions.cuh"

int main() {
    Boid *deviceBoids = nullptr;
    Boid *hostBoids = static_cast<Boid *>(malloc(N_BOIDS * sizeof(Boid)));
    float2 *deviceAccelerations = nullptr;
    curandState *deviceStates = nullptr;
    cudaError_t err = {};

    err = cudaMalloc(&deviceBoids, N_BOIDS * sizeof(Boid));
    if (err != cudaSuccess) {
        std:: cerr << "Fail to allocate memory for device boids\n";
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&deviceAccelerations, N_BOIDS * sizeof(float2));
    if (err != cudaSuccess) {
        std:: cerr << "Fail to allocate memory for device accelerations\n";
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&deviceStates, N_BOIDS * sizeof(curandState));
    if (err != cudaSuccess) {
        std:: cerr << "Fail to allocate memory for device states\n";
        exit(EXIT_FAILURE);
    }

    kernelLoadBoids<<<BLOCKS, TPB>>>(deviceBoids, deviceStates, 123ULL);
    cudaDeviceSynchronize();

    float currentTime = 0.0f;
    while (currentTime <= MAX_TIME) {
        kernelMakeBoidAcceleration<<<BLOCKS, TPB>>>(deviceBoids, deviceAccelerations);
        cudaDeviceSynchronize();

        kernelIntegrateBoids<<<BLOCKS, TPB>>>(deviceBoids, deviceAccelerations);
        cudaDeviceSynchronize();

        cudaMemcpy(hostBoids, deviceBoids, N_BOIDS * sizeof(Boid), cudaMemcpyDeviceToHost);

         for (int i = 0; i < N_BOIDS; i++) {
             printf("Boid at index %d\n", i);
             printf("Position (%.6f, %.2f)\n", hostBoids[i].position.x, hostBoids[i].position.y);
             printf("Velocity (%.6f, %.2f)\n", hostBoids[i].velocity.x, hostBoids[i].velocity.y);
             printf("\n");
         }

        currentTime += DT;
    }

    cudaFree(deviceBoids);
    cudaFree(deviceAccelerations);
    cudaFree(deviceStates);

    return 0;
}
