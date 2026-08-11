//
// Created by elder on 8/10/2026.
//

#ifndef CUDA_BOID_DEVICEFUNCTIONS_H
#define CUDA_BOID_DEVICEFUNCTIONS_H

constexpr int N_BOIDS = 4;
constexpr int TPB = 16;
constexpr int BLOCKS = (N_BOIDS + TPB - 1) / TPB;
constexpr float DT = 0.0016;
constexpr float MAX_TIME = 10.0f;
constexpr float SEPARATION_WEIGHT = 1.5f;
constexpr float ALIGNMENT_WEIGHT = 1.0f;
constexpr float COHESION_WEIGHT = 1.0f;
constexpr float PERCEPTION_RADIUS = 15.0f;
constexpr float SEPARATION_RADIUS = 5.0f;


struct Boid {
    bool inPerceptionRadius = false;
    bool inSeparationRadius = false;
    float mass = 0.0f;
    float2 position = {};
    float2 velocity = {};
};

inline __host__ __device__ float2 operator -(const float2 &a, const float2 &b) {
    return {a.x - b.x, a.y - b.y};
}

__device__ float2 kernelSeparationAverage(float2 *awayVectorsIn, const int validBoids);

__device__ float2 kernelAlignment(Boid *boids, const int validBoids);

__device__ float2 kernelCohesion(Boid *boids, const int validBoids);

__device__ float kernelDistance(const Boid &boidSelf, const Boid &boidNeighbor);

__device__ float2 kernelAwayVector(Boid &boidSelf, Boid &boidNeighbor);

__global__ void kernelFindNeighbors(Boid *boids, float2 *awayVectors);



#endif //CUDA_BOID_DEVICEFUNCTIONS_H