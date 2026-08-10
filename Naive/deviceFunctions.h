//
// Created by elder on 8/10/2026.
//

#ifndef CUDA_BOID_DEVICEFUNCTIONS_H
#define CUDA_BOID_DEVICEFUNCTIONS_H

constexpr int N_BOIDS = 100000;
constexpr int TPB = 256;
constexpr int BLOCKS = (N_BOIDS + TPB - 1) / TPB;
constexpr float DT = 0.0016;
constexpr float MAX_TIME = 10.0f;
constexpr float SEPARATION_WEIGHT = 1.5f;
constexpr float ALIGNMENT_WEIGHT = 1.0f;
constexpr float COHESION_WEIGHT = 1.0f;
constexpr float PERCEPTION_RADIUS = 15.0f;
constexpr float SEPARATION_RADIUS = 5.0f;


struct Boid {
    float2 position;
    float2 velocity;
    float mass;
    bool inPerceptionRadius = false;
    bool inSeparationRadius = false;
};

__device__ float kernelDistance(Boid &boid) {
    return sqrtf(boid.position.x * boid.position.x + boid.position.y * boid.position.y);
}

__global__ void kernelFindNeighbors(Boid *boids) {
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex >= N_BOIDS) return;

    for (int i = 0; i < N_BOIDS; i++) {
        if (globalIndex == i) continue;

        float distance = kernelDistance(boids[i]);

        if (distance <= PERCEPTION_RADIUS) boids[i].inPerceptionRadius = true;
        if (distance <= SEPARATION_RADIUS) boids[i].inSeparationRadius = true;
    }
}



#endif //CUDA_BOID_DEVICEFUNCTIONS_H