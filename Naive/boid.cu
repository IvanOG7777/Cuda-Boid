//
// Created by elder on 8/10/2026.
//

#include <cassert>
#include <iostream>
#include "deviceFunctions.h"

int main() {
    Boid *boids = static_cast<Boid *>(malloc(N_BOIDS * sizeof(Boid)));

    assert(boids != nullptr);

    Boid *deviceBoids = nullptr;
    float2 *deviceAverageOut = nullptr;
    int *deviceValidBoids = nullptr;
    float2 *deviceAcceleration = nullptr;
    curandState *states = nullptr;

    cudaMalloc(&deviceBoids, N_BOIDS * sizeof(Boid));
    cudaMalloc(&deviceAverageOut, sizeof(float2));
    cudaMalloc(&deviceValidBoids, sizeof(int));
    cudaMalloc(&deviceAcceleration, N_BOIDS * sizeof(float2));
    cudaMalloc(&states, N_BOIDS * sizeof(curandState));

    kernelLoadBoids<<<BLOCKS, TPB>>>(deviceBoids, states, 123ULL);
    cudaDeviceSynchronize();


    float currentTime = 0.0f;
    while (currentTime <= MAX_TIME) {

        kernelRunBoids<<<BLOCKS, TPB>>>(deviceBoids, deviceAcceleration);
        cudaDeviceSynchronize();
        kernelIntegrateBoid<<<BLOCKS, TPB>>>(deviceBoids, deviceAcceleration);

        cudaMemcpy(boids, deviceBoids, N_BOIDS * sizeof(Boid), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N_BOIDS; i++) {
            printf("Boid at index: %d\n", i);
            printf("Position: (%.6f, %.6f)\n", boids[i].position.x, boids[i].position.y);
            printf("Velocity: (%.6f, %.6f)\n", boids[i].velocity.x, boids[i].velocity.y);
            printf("\n");
        }
        currentTime += DT;
    }

    return 0;
}