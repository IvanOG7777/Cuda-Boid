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

__device__ float2 kernelMakeAwayVector(Boid &boidSelf, Boid &boidNeighbor) {
    float2 resultingVector = {};

    resultingVector = boidSelf.position - boidNeighbor.position;

    return resultingVector;
}

__device__ float2 kernelSeparationAverage(float2 *awayVectorsIn, const int validBoids) {
    if (validBoids == 0) return {0.0f, 0.0f};
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndex >= N_BOIDS) return {0.0f, 0.0f};

    float x = 0.0f;
    float y = 0.0f;

    // Loop through all boids but invalid vectors at i would be {0,0}, doesnt change anything
    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        x += awayVectorsIn[i].x;
        y += awayVectorsIn[i].y;
    }

    x /= static_cast<float>(validBoids);
    y /= static_cast<float>(validBoids);

    return {x, y};
}

//Takes in array of valid boids and their count
// When entering this function all of *boids should be valid
__device__ float2 kernelAlignment(Boid *boids, int validBoids) {
    if (validBoids == 0) return {0.0f, 0.0f};
    float x = 0.0f;
    float y = 0.0f;

    float2 alignment = {};

    for (int i = 0; i < validBoids; i++) {

        x += boids[i].velocity.x;
        y += boids[i].velocity.y;

    }

    x /= static_cast<float>(validBoids);
    y /= static_cast<float>(validBoids);

    alignment.x = x;
    alignment.y = y;

    return alignment;
}

//Takes in array of valid boids and their count
// When entering this function all of *boids should be valid
__device__ float2 kernelCohesion(Boid *boids, const int validBoids) {
    if (validBoids == 0) return {0.0f, 0.0f};
    float x = 0.0f;
    float y = 0.0f;

    float2 cohesion = {0.0f, 0.0f};

    // loop though validBoids
    for (int i = 0; i < validBoids; i++) {
        x += boids[i].position.x;
        y += boids[i].position.y;
    }

    x /= static_cast<float>(validBoids);
    y /= static_cast<float>(validBoids);

    cohesion.x = x;
    cohesion.y = y;

    return cohesion;
}

__device__ float2 kernelMakeBoidAcceleration(const float2 separation, const float2 alignment, const float2 cohesion) {
    float2 acceleration = {};

    acceleration.x = (SEPARATION_WEIGHT * separation.x) + (ALIGNMENT_WEIGHT * alignment.x) + (COHESION_WEIGHT * cohesion.x);
    acceleration.y = (SEPARATION_WEIGHT * separation.y) + (ALIGNMENT_WEIGHT * alignment.y) + (COHESION_WEIGHT * cohesion.y);

    return acceleration;
}

__global__ void kernelRunBoids(Boid *boids, float2 *accelerationOut) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

     int boidsWithinPerception = 0;
     int boidsWithinSeparation = 0;

    float2 localAwayVector[N_BOIDS] = {};
    Boid perceptionBoids[N_BOIDS] = {};

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        float distance = kernelDistance(boids[globalIndex], boids[i]);

        if (distance <= PERCEPTION_RADIUS) {
            boidsWithinPerception++;
        }
        if (distance <= SEPARATION_RADIUS) {
            localAwayVector[i] = kernelAwayVector(boids[globalIndex], boids[i]);
            boidsWithinSeparation++;
        } else {
            localAwayVector[i] = {0,0};
        }
    }

    int perceptionCount = 0;

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;
        float distance = kernelDistance(boids[globalIndex], boids[i]);

        if (distance <= PERCEPTION_RADIUS) {
            perceptionBoids[perceptionCount++] = boids[i]; // by this time will only hold boidsWithinPerception valid boids
        }
    }

    float2 averageSeparation = kernelSeparationAverage(localAwayVector, boidsWithinSeparation);

    float2 alignment = kernelAlignment(perceptionBoids, boidsWithinPerception);

    alignment = alignment - boids[globalIndex].velocity;

    float2 cohesion = kernelCohesion(perceptionBoids, boidsWithinPerception);

    cohesion = cohesion - boids[globalIndex].position;

    float2 acceleration = kernelMakeBoidAcceleration(averageSeparation, alignment, cohesion);

    accelerationOut[globalIndex] = acceleration;
}

__global__ void kernelIntegrateBoid(Boid *boid, const float2 *acceleration) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndex >= N_BOIDS) return;
    float2 newVelocity = {};
    float2 newPosition = {};

    newVelocity = boid[globalIndex].velocity + (acceleration[globalIndex] * DT);
    newPosition = boid[globalIndex].position + (newVelocity * DT);

    boid[globalIndex].velocity = newVelocity;
    boid[globalIndex].position = newPosition;
}

__device__ void kernelInitState(curandState *states, const unsigned int seed) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    curand_init(seed, globalIndex, 0, &states[globalIndex]);
}
__device__ float2 kernelRandFloat2(curandState *state) {
    float2 rand = {};

    rand.x = curand_uniform(state);
    rand.y = curand_uniform(state);

    return rand;
}
__global__ void kernelLoadBoids(Boid *boids, curandState *states, const unsigned int seed) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    kernelInitState(states, seed);

    boids[globalIndex].position = kernelRandFloat2(&states[globalIndex]);
    boids[globalIndex].velocity = kernelRandFloat2(&states[globalIndex]);
}