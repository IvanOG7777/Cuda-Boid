//
// Created by elder on 8/10/2026.
//

#include <cassert>
#include <iostream>
#include "deviceFunctions.h"

int main() {

    Boid *boids = static_cast<Boid *>(malloc(N_BOIDS * sizeof(Boid)));
    float2 *awayVectors = static_cast<float2 *>(malloc(N_BOIDS * sizeof(float2)));
    float2 *averageHost = static_cast<float2 *>(malloc(sizeof(float)));
    int *validBoids = static_cast<int *>(malloc(sizeof(int)));

    assert(boids != nullptr);
    assert(awayVectors != nullptr);

    boids[0].position = {0.0f, 0.0f};
    boids[0].velocity = {1.0f, 0.0f};

    boids[1].position = {2.0f, 0.0f};
    boids[1].velocity = {0.0f, 1.0f};

    boids[2].position = {1.0f, 1.0f};
    boids[2].velocity = {1.0f, 1.0f};

    boids[3].position = {-1.0f, 1.0f};
    boids[3].velocity = {0.5f, 0.5f};

    for (int i = 4; i < N_BOIDS; i++) {
        boids[i] = {};
    }

    for (int i = 0; i < N_BOIDS; i++) {
        awayVectors[i] = {0.0f, 0.0f};
    }

    Boid *deviceBoids = nullptr;
    float2 *deviceAwayVectors = nullptr;
    float2 *deviceAverageOut = nullptr;
    int *deviceValidBoids = nullptr;

    cudaMalloc(&deviceBoids, N_BOIDS * sizeof(Boid));
    cudaMalloc(&deviceAwayVectors, N_BOIDS * sizeof(float2));
    cudaMalloc(&deviceAverageOut, sizeof(float2));
    cudaMalloc(&deviceValidBoids, sizeof(int));

    cudaMemcpy(deviceBoids, boids, N_BOIDS * sizeof(Boid), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceAwayVectors, awayVectors, N_BOIDS * sizeof(float2), cudaMemcpyHostToDevice);


    kernelFindNeighbors<<<BLOCKS, TPB>>>(deviceBoids, deviceAwayVectors, deviceValidBoids);
    cudaDeviceSynchronize();
    kernelAwayAverage<<<BLOCKS, TPB>>>(deviceAwayVectors, deviceValidBoids, deviceAverageOut);
    cudaDeviceSynchronize();

    cudaMemcpy(awayVectors, deviceAwayVectors, N_BOIDS * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(averageHost, deviceAverageOut, sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(validBoids, deviceValidBoids, sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_BOIDS; i++) {
        printf("(%.2f, %.2f)\n", awayVectors[i].x, awayVectors[i].y);
    }

    std:: cout << std:: endl;
    std:: cout << std:: endl;

    // Wrong average and boid count
    std:: cout << "Average separation is: " << averageHost->x << ", " << averageHost->y << std:: endl;
    std:: cout << "Valid boid count is: " << *validBoids << std:: endl;


    return 0;
}