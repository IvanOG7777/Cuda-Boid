//
// Created by elder on 8/10/2026.
//

#include "deviceFunctions.h"

__device__ float kernelDistance(Boid &boid) {
    return sqrtf(boid.position.x * boid.position.x + boid.position.y * boid.position.y);
}

__device__ float2 kernelAwayVector(Boid &boidSelf, Boid &boidNeighbor) {
    float2 resultingVector = {};

    resultingVector = boidSelf.position - boidNeighbor.position;

    return resultingVector;
}

__device__ float2 kernelAwayAverage(float2 *awayVectors, unsigned int globalIndex, int validBoids) {
    float x = 0.0f;
    float y = 0.0f;
    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        x += awayVectors[i].x;
        y += awayVectors[i].y;
    }

    x /= static_cast<float>(validBoids);
    y /= static_cast<float>(validBoids);

    return {x, y};
}

__global__ void kernelFindNeighbors(Boid *boids, float2 *awayVectors) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    int boidsWithinPerception = 0;
    int boidsWithinSeparation = 0;

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        float distance = kernelDistance(boids[i]);

        // Valid Neighbors
        if (distance <= PERCEPTION_RADIUS) boids[i].inPerceptionRadius = true, boidsWithinPerception++;
        if (distance <= SEPARATION_RADIUS) boids[i].inSeparationRadius = true, boidsWithinSeparation++;

        // Too far
        if (distance > PERCEPTION_RADIUS) boids[i].inPerceptionRadius = false;
        if (distance > SEPARATION_RADIUS) boids[i].inSeparationRadius = false;
    }

    // calculate each away vector within separation radius
    for (int i = 0; i < N_BOIDS; i++) {
        if (i == globalIndex) continue;

        if (boids[i].inSeparationRadius) {
            awayVectors[i] = kernelAwayVector(boids[globalIndex], boids[i]);
        } else {
            awayVectors[i] = {0,0};
        }
    }
}