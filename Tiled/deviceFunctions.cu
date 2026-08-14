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

__device__ float2 kernelCalculateAcceleration(float2 averageSeparation, float2 averageAlignment, float2 averageCohesion) {
    float2 acceleration = {0.0f, 0.0f};

    acceleration.x = SEPARATION_WEIGHT * averageSeparation.x + ALIGNMENT_WEIGHT * averageAlignment.x + COHESION_WEIGHT * averageCohesion.x;
    acceleration.y = SEPARATION_WEIGHT * averageSeparation.y + ALIGNMENT_WEIGHT * averageAlignment.y + COHESION_WEIGHT * averageCohesion.y;

    return acceleration;
}

__global__ void kernelMakeBoidAcceleration(Boid *boids, float2 *accelerationOut) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localThread = threadIdx.x;

    // accumulators to be used after tile/block loop
    float2 globalSeparationSum = {0.0f, 0.0f};
    float2 globalAlignmentSum = {0.0f, 0.0f};
    float2 globalCohesionSum = {0.0f, 0.0f};
    int globalBoidsWithinPerception = 0;
    int globalBoidsWithinSeparation = 0;

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

        int boidsWithinPerception = 0;
        int boidsWithinSeparation = 0;

        // loop trough current loaded blocks tiles
        for (int i = 0; i < TPB; i++) {

            unsigned threadIndex = i + TPB * tile;

            if (globalIndex >= N_BOIDS) continue;
            if (globalIndex == threadIndex) continue;
            if (blockBoids[i].valid == false) continue;

            float distance = kernelDistanceBoidAB(boids[globalIndex], blockBoids[i]);

            if (distance <= PERCEPTION_RADIUS) {
                globalAlignmentSum += blockBoids[i].velocity;
                globalCohesionSum += blockBoids[i].position;
                boidsWithinPerception++;
            }

            if (distance <= SEPARATION_RADIUS) {
                boidsWithinSeparation++;
                float2 awayVector = kernelCalculateAwayVector(boids[globalIndex], blockBoids[i]);
                globalSeparationSum += awayVector;
            }
        }
        __syncthreads(); // wait for threads to finish before accumulation

        globalBoidsWithinPerception += boidsWithinPerception;
        globalBoidsWithinSeparation += boidsWithinSeparation;
    }

    if (globalIndex < N_BOIDS) {
        // create partial acceleration if separation or perception returns 0
        if (globalBoidsWithinSeparation == 0 && globalBoidsWithinPerception != 0) {
            float2 averageAlignment = globalAlignmentSum / globalBoidsWithinPerception;
            float2 averageCohesion = globalCohesionSum / globalBoidsWithinPerception;

            averageAlignment = averageAlignment - boids[globalIndex].velocity;

            averageCohesion = averageCohesion - boids[globalIndex].position;

            float2 acceleration = kernelCalculateAcceleration({0.0f, 0.0f}, averageAlignment, averageCohesion);

            accelerationOut[globalIndex] = acceleration;

            return;
        }

        if (globalBoidsWithinPerception == 0 && globalBoidsWithinSeparation != 0) {
            float2 averageSeparation = globalSeparationSum / globalBoidsWithinSeparation;

            float2 acceleration = kernelCalculateAcceleration(averageSeparation, {0.0f, 0.0f}, {0.0f, 0.0f});

            accelerationOut[globalIndex] = acceleration;

            return;
        }

        if (globalBoidsWithinPerception == 0 && globalBoidsWithinSeparation == 0) {
            accelerationOut[globalIndex] = {0.0f, 0.0f};

            return;
        }
        ////

        // if we have all values make final acceleration
        float2 averageSeparation = globalSeparationSum / globalBoidsWithinSeparation;
        float2 averageAlignment = globalAlignmentSum / globalBoidsWithinPerception;
        float2 averageCohesion = globalCohesionSum / globalBoidsWithinPerception;

        averageAlignment = averageAlignment - boids[globalIndex].velocity;
        averageCohesion = averageCohesion - boids[globalIndex].position;

        float2 acceleration = kernelCalculateAcceleration(averageSeparation, averageAlignment, averageCohesion);

        accelerationOut[globalIndex] = acceleration;
    }
}

__global__ void kernelIntegrateBoids(Boid *boids, float2 *accelerationIn) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    float2 newPosition = {0.0f, 0.0f};
    float2 newVelocity = {0.0f, 0.0f};

    newVelocity = boids[globalIndex].velocity + (accelerationIn[globalIndex] * DT);
    newPosition = boids[globalIndex].position + (newVelocity * DT);

    boids[globalIndex].position = newPosition;
    boids[globalIndex].velocity = newVelocity;
}

__device__ void kernelInitStates(curandState *states, unsigned int seed, unsigned int index) {
    curand_init(seed, index, 0, &states[index]);
}

__device__ float2 kernelRandFloat2(curandState *state) {
    float2 rand = {0.0f, 0.0f};

    rand.x = curand_uniform(state) * 10;
    rand.y = curand_uniform(state) * 10;

    return rand;
}

__global__ void kernelLoadBoids(Boid *boids, curandState *states, const unsigned int seed) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    kernelInitStates(states, seed, globalIndex);

    boids[globalIndex].position = kernelRandFloat2(&states[globalIndex]);
    boids[globalIndex].velocity = kernelRandFloat2(&states[globalIndex]);
    boids[globalIndex].valid = true;
}