//
// Created by elder on 8/10/2026.
//

#include <cassert>
#include <iostream>
#include "deviceFunctions.h"

int main() {
    Boid *boids = static_cast<Boid *>(malloc(N_BOIDS * sizeof(Boid)));

    assert(boids != nullptr);

    boids[0].position = {0.0f, 0.0f};
    boids[0].velocity = {1.0f, 0.0f};

    boids[1].position = {2.0f, 0.0f};
    boids[1].velocity = {0.0f, 1.0f};

    boids[2].position = {1.0f, 1.0f};
    boids[2].velocity = {1.0f, 1.0f};

    boids[3].position = {-1.0f, 1.0f};
    boids[3].velocity = {0.5f, 0.5f};

    Boid *deviceBoids = nullptr;
    float2 *deviceAverageOut = nullptr;
    int *deviceValidBoids = nullptr;
    float2 *deviceAcceleration = nullptr;

    cudaMalloc(&deviceBoids, N_BOIDS * sizeof(Boid));
    cudaMalloc(&deviceAverageOut, sizeof(float2));
    cudaMalloc(&deviceValidBoids, sizeof(int));
    cudaMalloc(&deviceAcceleration, N_BOIDS * sizeof(float2));

    cudaMemcpy(deviceBoids, boids, N_BOIDS * sizeof(Boid), cudaMemcpyHostToDevice);


    float currentTime = 0.0f;
    while (currentTime <= MAX_TIME) {

        kernelRunBoids<<<BLOCKS, TPB>>>(deviceBoids, deviceAcceleration);
        cudaDeviceSynchronize();
        kernelIntegrateBoid<<<BLOCKS, TPB>>>(deviceBoids, deviceAcceleration);

        cudaMemcpy(boids, deviceBoids, N_BOIDS * sizeof(Boid), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N_BOIDS; i++) {
            printf("Boid at index: %d\n", i);
            printf("Position: (%.2f, %.2f)\n", boids[i].position.x, boids[i].position.y);
            printf("Velocity: (%.2f, %.2f)\n", boids[i].velocity.x, boids[i].velocity.y);
            printf("\n");
        }
        currentTime += DT;
    }

    return 0;
}