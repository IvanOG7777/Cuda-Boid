//
// Created by elder on 8/12/2026.
//

#include "deviceFunctions.cuh"

__device__ float kernelDistanceBoidAB(const Boid &boidSelf, const Boid &boidNeighbor) {
    if (boidSelf.valid == false || boidNeighbor.valid == false) return 0.0f;
    float x = boidNeighbor.position.x - boidSelf.position.x;
    float y = boidNeighbor.position.y - boidSelf.position.y;

    return sqrtf(x * x + y * y);
}

__device__ float2 kernelCalculateAwayVector(Boid &boidSelf, Boid &boidNeighbor) {
    if (boidSelf.valid == false || boidNeighbor.valid == false) return {0.0f, 0.0f};
    return boidSelf.position - boidNeighbor.position;
}

// when passing validAwayVector make sure its size if of validBoidCount
// We are assuming here that all values passed are valid
/////
__device__ float2 kernelSeparationAverage(const float2 *validAwayVectors, const int validBoidCount) {
    if (validBoidCount == 0) return {0.0f};
    float2 average = {0.0f, 0.0f};
    for (int i = 0; i < validBoidCount; i++) {
        average.x += validAwayVectors[i].x;
        average.y += validAwayVectors[i].y;
    }

    average.x /= static_cast<float>(validBoidCount);
    average.y /= static_cast<float>(validBoidCount);

    return average;
}

__device__ float2 kernelAlignmentAverage(const Boid *validBoids, int validBoidCount) {
    if (validBoidCount == 0) return {0.0f};

    float2 alignment = {0.0f, 0.0f};

    for (int i = 0; i < validBoidCount; i++) {
        alignment.x += validBoids[i].velocity.x;
        alignment.y += validBoids[i].velocity.y;
    }

    alignment.x /= static_cast<float>(validBoidCount);
    alignment.y /= static_cast<float>(validBoidCount);

    return alignment;
}

__device__ float2 kernelCohesionAverage(const Boid *validBoids, int validBoidCount) {
    if (validBoidCount == 0) return {0.0f, 0.0f};

    float2 cohesion = {0.0f, 0.0f};

    for (int i = 0; i < validBoidCount; i++) {
        cohesion.x += validBoids[i].position.x;
        cohesion.y += validBoids[i].position.y;
    }

    cohesion.x /= static_cast<float>(validBoidCount);
    cohesion.y /= static_cast<float>(validBoidCount);

    return cohesion;
}

/////

__device__ float2
kernelCalculateAcceleration(float2 averageSeparation, float2 averageAlignment, float2 averageCohesion) {
    float2 acceleration = {0.0f, 0.0f};

    acceleration.x = SEPARATION_WEIGHT * averageSeparation.x + ALIGNMENT_WEIGHT * averageAlignment.x + COHESION_WEIGHT * averageCohesion.x;
    acceleration.y = SEPARATION_WEIGHT * averageSeparation.y + ALIGNMENT_WEIGHT * averageAlignment.y + COHESION_WEIGHT * averageCohesion.y;

    return acceleration;
}

__global__ void kernelMakeBoidAcceleration(Boid *boids, float2 *accelerationOut) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localThread = threadIdx.x;

    // loop through blocks
    for (int tile = 0; tile < BLOCKS; tile++) {

        unsigned int tileIndex = localThread + TPB * tile; // get current tile index within block

        __shared__ Boid blockBoids[TPB]; // create a shared array of boids for current block. (All threads of this block can see it)

        // load boids from global to local block memory
        if (tileIndex < N_BOIDS) {
            blockBoids[localThread] = boids[tileIndex];
            blockBoids[localThread].valid = true; // only change local block state of boid to true
        } else {
            blockBoids[localThread] = Boid{};
        }
        __syncthreads(); // wait for threads to finish in order to use blockBoids

        float2 awayVectors[TPB] = {};
        Boid validBoids[TPB] = {};

        int boidsWithinPerception = 0;
        int boidsWithinSeparation = 0;

        int awayVectorIndex = 0;
        int validBoidsIndex = 0;

        for (int i = 0; i < TPB; i++) {
            unsigned threadIndex = i + TPB * tile;
            if (globalIndex == threadIndex) continue;
            if (blockBoids[i].valid == false) continue;

            float distance = kernelDistanceBoidAB(boids[globalIndex], blockBoids[i]);

            if (distance <= PERCEPTION_RADIUS) {
                boidsWithinPerception++;
            }

            if (distance <= SEPARATION_RADIUS) {
                boidsWithinSeparation++;
                awayVectors[awayVectorIndex++] = kernelCalculateAwayVector(boids[globalIndex], blockBoids[i]);
            }
        }
        __syncthreads(); // wait for threads to finish in order to use awayVectors

        for (int i = 0; i < TPB; i++) {
            unsigned threadIndex = i + TPB * tile;
            if (globalIndex == threadIndex) continue;
            if (blockBoids[i].valid == false) continue;
            float distance = kernelDistanceBoidAB(boids[globalIndex], blockBoids[i]);

            if (distance <= PERCEPTION_RADIUS) {
                validBoids[validBoidsIndex++] = blockBoids[i];
            }
        }
        __syncthreads(); // wait for thread to finish in order to use validBoids

        // calculate local average sums
        float2 averageSeparation = kernelSeparationAverage(awayVectors, boidsWithinSeparation);
        float2 averageAlignment = kernelAlignmentAverage(validBoids, boidsWithinPerception);
        float2 averageCohesion = kernelCohesionAverage(validBoids, boidsWithinPerception);

        averageAlignment = averageAlignment - boids[globalIndex].velocity;
        averageCohesion = averageCohesion - boids[globalIndex].position;

        float2 acceleration = kernelCalculateAcceleration(averageSeparation, averageAlignment, averageCohesion);

        accelerationOut[globalIndex] = acceleration;
    }
}

__global__ void integrateBoids(Boid *boids, const float2 *accelerationIn) {
}
