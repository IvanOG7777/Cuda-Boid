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

__device__ float2 kernelSeparationAverage(float2 *awayVectorsIn, const int validBoids) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndex >= N_BOIDS) return {0.0f, 0.0f};

    float x = 0.0f;
    float y = 0.0f;

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        x += awayVectorsIn[i].x;
        y += awayVectorsIn[i].y;
    }

    x /= static_cast<float>(validBoids);
    y /= static_cast<float>(validBoids);

    return {x, y};
}

__device__ float2 kernelAlignment(Boid *boids, int validBoids) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return {};

    float x = 0.0f;
    float y = 0.0f;

    float2 alignment = {};

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        x += boids[i].velocity.x;
        y += boids[i].velocity.y;

    }

    x /= static_cast<float>(validBoids);
    y /= static_cast<float>(validBoids);

    alignment.x = x;
    alignment.y = y;

    return alignment;
}

__device__ float2 kernelCohesion(Boid *boids, const int validBoids) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return {};

    float x = 0.0f;
    float y = 0.0f;

    float2 cohesion = {0.0f, 0.0f};

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        x += boids[i].position.x;
        y += boids[i].position.y;
    }

    x /= static_cast<float>(validBoids);
    y /= static_cast<float>(validBoids);

    cohesion.x = x;
    cohesion.y = y;

    return cohesion;
}

__global__ void kernelFindNeighbors(Boid *boids, float2 *awayVectors) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

     int boidsWithinPerception = 0;
     int boidsWithinSeparation = 0;

    float2 localAwayVector[N_BOIDS];

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        float distance = kernelDistance(boids[globalIndex], boids[i]);

        if (distance <= PERCEPTION_RADIUS) {
            boidsWithinPerception++;
        }
        if (distance <= SEPARATION_RADIUS) {
            boidsWithinSeparation++;
        }

    }
    __syncthreads();

    // calculate each away vector within separation radius
    for (int i = 0; i < N_BOIDS; i++) {
        if (i == globalIndex) continue;

        float distance = kernelDistance(boids[globalIndex], boids[i]);

        if (distance <= SEPARATION_RADIUS) {
            localAwayVector[i] = kernelAwayVector(boids[globalIndex], boids[i]);
        } else {
            localAwayVector[i] = {0,0};
        }
    }
    __syncthreads();

    float2 averageSeparation = kernelSeparationAverage(localAwayVector, boidsWithinSeparation);

    printf("Average separation for thread %d is: (%.2f, %.2f)\n", globalIndex, averageSeparation.x, averageSeparation.y);

    float2 alignment = kernelAlignment(boids, boidsWithinSeparation);

    alignment = alignment - boids[globalIndex].velocity;

    printf("Average alignment for thread %d is: (%.2f, %.2f)\n", globalIndex, alignment.x, alignment.y);

    float2 cohesion = kernelCohesion(boids, boidsWithinSeparation);

    cohesion = cohesion - boids[globalIndex].position;

    printf("Average cohesion for thread %d is: (%.2f, %.2f)\n", globalIndex, cohesion.x, cohesion.y);

    awayVectors[globalIndex] = averageSeparation;
}