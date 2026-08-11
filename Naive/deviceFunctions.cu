//
// Created by elder on 8/10/2026.
//

#include "deviceFunctions.h"

#include <cstdio>

__device__ float kernelDistance(const Boid &boidSelf, const Boid &boidNeighbor) {
    float x = boidNeighbor.position.x - boidSelf.position.x;
    float y = boidNeighbor.position.y - boidSelf.position.y;
    return sqrtf(x * x + y * y);
}

__device__ float2 kernelAwayVector(Boid &boidSelf, Boid &boidNeighbor) {
    float2 resultingVector = {};

    resultingVector = boidSelf.position - boidNeighbor.position;

    return resultingVector;
}

__global__ void kernelAwayAverage(float2 *awayVectorsIn, int *validBoids, float2 *averageOut) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndex >= N_BOIDS) return;

    float x = 0.0f;
    float y = 0.0f;
    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        x += awayVectorsIn[i].x;
        y += awayVectorsIn[i].y;
    }

    x /= static_cast<float>(validBoids[0]);
    y /= static_cast<float>(validBoids[0]);

    averageOut->x = x;
    averageOut->y = y;
}

__global__ void kernelFindNeighbors(Boid *boids, float2 *awayVectors, int *validBoids) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    int boidsWithinPerception = 0;
    int boidsWithinSeparation = 0;

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        float distance = kernelDistance(boids[globalIndex], boids[i]);
        printf("Distance: %.2f\n", distance);

        // Valid Neighbors
        //TODO: With 4 valid values on each loop and thread its recounting already valid boids. Valid boids should be 3 but im getting 9
        // maybe add another flag
        if (distance <= PERCEPTION_RADIUS) boids[i].inPerceptionRadius = true, boidsWithinPerception++;
        if (distance <= SEPARATION_RADIUS) boids[i].inSeparationRadius = true, boidsWithinSeparation++;

        // Too far
        if (distance > PERCEPTION_RADIUS) boids[i].inPerceptionRadius = false;
        if (distance > SEPARATION_RADIUS) boids[i].inSeparationRadius = false;
    }
    __syncthreads();

    validBoids[0] = boidsWithinSeparation;

    // calculate each away vector within separation radius
    for (int i = 0; i < N_BOIDS; i++) {
        if (i == globalIndex) continue;

        if (boids[i].inSeparationRadius == true) {
            awayVectors[i] = kernelAwayVector(boids[globalIndex], boids[i]);
        } else {
            awayVectors[i] = {0,0};
        }
    }
    __syncthreads();
}