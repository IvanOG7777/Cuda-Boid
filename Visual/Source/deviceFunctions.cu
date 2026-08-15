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

__device__ float2 kernelCalculateAcceleration(const float2 &separation, const float2 &alignment, const float2 &cohesion) {
    float2 acceleration = {0.0f, 0.0f};

    acceleration.x = SEPARATION_WEIGHT * separation.x + ALIGNMENT_WEIGHT * alignment.x + COHESION_WEIGHT * cohesion.x;
    acceleration.y = SEPARATION_WEIGHT * separation.y + ALIGNMENT_WEIGHT * alignment.y + COHESION_WEIGHT * cohesion.y;

    return acceleration;
}

__global__ void kernelMakeAcceleration(Boid *boids, float2 *accelerationsOut) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localThread = threadIdx.x;

    float2 separationSum = {0.0f, 0.0f};
    float2 alignmentSum = {0.0f, 0.0f};
    float2 cohesionSum = {0.0f, 0.0f};
    int totalBoidsWithinSeparation = 0;
    int totalBoidsWithinPerception = 0;

    for (int block = 0; block < BLOCKS; block++) {
        unsigned int tileIndex = localThread + TPB * block;

        __shared__ Boid blockBoids[TPB];

        if (tileIndex < N_BOIDS) {
            blockBoids[localThread] = boids[tileIndex];
            blockBoids[localThread].valid = true;
        } else {
            blockBoids[localThread] = Boid{};
        }
        __syncthreads();

        int boidsWithinPerception = 0;
        int boidsWithinSeparation = 0;

        for (int i = 0; i < TPB; i++) {
            unsigned tileBoidIndex = i + TPB * tileIndex; // used as sort of globalIndex but within blockBoids

            if (globalIndex == tileBoidIndex) continue;
            if (blockBoids[tileBoidIndex].valid == false) continue;
            if (globalIndex >= N_BOIDS) return;

            float distance = kernelDistanceAB(boids[globalIndex], blockBoids[i]);

            if (distance <= PERCEPTION_RADIUS && distance > 0.0f) {
                boidsWithinPerception++;
                alignmentSum += blockBoids[i].velocity;
                cohesionSum += blockBoids[i].position;
            }

            if (distance <= SEPARATION_RADIUS && distance > 0.0f) {
                boidsWithinSeparation++;
                float2 awayVector = kernelCalculateAwayVector(boids[globalIndex], blockBoids[i]);
                separationSum += awayVector;
            }
        }
        __syncthreads();

        totalBoidsWithinPerception += boidsWithinPerception;
        totalBoidsWithinSeparation += boidsWithinSeparation;
    }

    if (globalIndex < N_BOIDS) {

        if (totalBoidsWithinSeparation == 0 && totalBoidsWithinPerception != 0) { // only separation is 0
            float2 averageAlignment = alignmentSum / static_cast<float>(totalBoidsWithinPerception);
            float2 averageCohesion = cohesionSum / static_cast<float>(totalBoidsWithinPerception);

            float2 acceleration = kernelCalculateAcceleration({0.0f, 0.0f}, averageAlignment, averageCohesion);

            accelerationsOut[globalIndex] = acceleration;
        } else if (totalBoidsWithinSeparation != 0 && totalBoidsWithinPerception == 0) { // only perception is 0
            float2 averageSeparation = separationSum / static_cast<float>(totalBoidsWithinSeparation);

            float2 acceleration = kernelCalculateAcceleration(averageSeparation, {0.0f, 0.0}, {0.0f, 0.0});

            accelerationsOut[globalIndex] = acceleration;
        } else if (totalBoidsWithinSeparation == 0 && totalBoidsWithinPerception == 0) { // both are 0
            accelerationsOut[globalIndex] = {0.0f, 0.0f};
        } else { // both are valid non 0
            float2 averageAlignment = alignmentSum / static_cast<float>(totalBoidsWithinPerception);
            float2 averageCohesion = cohesionSum / static_cast<float>(totalBoidsWithinPerception);
            float2 averageSeparation = separationSum /static_cast<float>(totalBoidsWithinSeparation);

            float2 acceleration = kernelCalculateAcceleration(averageSeparation, averageAlignment, averageCohesion);

            accelerationsOut[globalIndex] = acceleration;
        }
    }
}

__global__ void integrateBoid(Boid *boids, float2 *accelerationsIn) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    float2 newPosition = {0.0f, 0.0f};
    float2 newVelocity = {0.0f, 0.0f};

    newVelocity = boids[globalIndex].velocity + accelerationsIn[globalIndex] * DT;
    newPosition = boids[globalIndex].position + newVelocity * DT;

    boids[globalIndex].position = newPosition;
    boids[globalIndex].velocity = newVelocity;
}

__device__ void kernelInitStates(curandState *states, unsigned int seed, unsigned int index) {
    curand_init(seed, index, 0, &states[index]);
}

__device__ float2 randFloat2(curandState *states, unsigned int index) {
    float2 rand = {0.0f, 0.0f};

    rand.x = curand_uniform(&states[index]);
    rand.y = curand_uniform(&states[index]);

    return rand;
}

__global__ void kernelInitBoids(Boid *boids, curandState *states, unsigned int seed) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    kernelInitStates(states, seed, globalIndex);

    float2 position = randFloat2(states, globalIndex);
    float2 velocity = randFloat2(states, globalIndex);

    boids[globalIndex].position = position;
    boids[globalIndex].velocity = velocity;
}