//
// Created by elder on 8/14/2026.
//

#include "../Header/deviceFunctions.cuh"

__device__ float kernelDistanceAB(const Boid &boidSelf, const Boid &boidNeighbor) {
    if (boidSelf.valid == false || boidNeighbor.valid == false) return 0.0f;
    float x = boidNeighbor.position.x - boidSelf.position.x;
    float y = boidNeighbor.position.y - boidSelf.position.y;

    return sqrtf(x * x + y * y);
}

__device__ float2 kernelCalculateAwayVector(Boid &boidSelf, Boid &boidNeighbor) {
    if (boidSelf.valid == false || boidNeighbor.valid == false) return {0.0f, 0.0f};

    return boidSelf.position - boidNeighbor.position;
}

__global__ void kernelMakeAcceleration(Boid *boids, float2 *accelerationsIn) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localThread = threadIdx.x;

    float2 separationSum = {0.0f, 0.0f};
    float2 accelerationSum = {0.0f, 0.0f};
    float2 cohesionSum = {0.0f, 0.0f};
    int totalBoidsWithinSeparation = 0;
    int totalBoidsWithinPerception = 0;

    for (int block = 0; block < BLOCKS; block++) {
        unsigned int tile = localThread + TPB * block;

        __shared__ Boid blockBoids[TPB];

        if (tile < N_BOIDS) {
            blockBoids[localThread] = boids[tile];
            blockBoids[localThread].valid = true;
        } else {
            blockBoids[localThread] = Boid{};
        }
        __syncthreads();

        int boidsWithinPerception = 0;
        int boidsWithinSeparation = 0;

        for (int i = 0; i < TPB; i++) {
            unsigned threadIndex = i + TPB * tile;
            float distance = kernelDistanceAB(boids[globalIndex], blockBoids[i]);
        }
    }
}