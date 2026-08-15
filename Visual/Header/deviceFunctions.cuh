//
// Created by elder on 8/14/2026.
//

#ifndef CUDAPRACTICE_DEVICEFUNCTIONS_CUH
#define CUDAPRACTICE_DEVICEFUNCTIONS_CUH

#include <curand_kernel.h>

constexpr int N_BOIDS = 100;
constexpr int TPB = 64;
constexpr int BLOCKS = (N_BOIDS + TPB - 1) / TPB;
constexpr float MAX_TIME = 10.0f;
constexpr float DT = 0.016f;
constexpr float SEPARATION_WEIGHT = 1.5f;
constexpr float ALIGNMENT_WEIGHT = 1.0f;
constexpr float COHESION_WEIGHT = 1.0f;
constexpr float PERCEPTION_RADIUS = 15.0f;
constexpr float SEPARATION_RADIUS = 5.0f;

struct Boid {
    bool valid = false;
    float2 position = {};
    float2 velocity = {};
};

inline __host__ __device__ float2 operator +(const float2 &a, const float2 &b) {
    return {a.x + b.x, a.y + b.y};
}

inline __host__ __device__ float2 operator -(const float2 &a, const float2 &b) {
    return {a.x - b.x, a.y - b.y};
}

inline __host__ __device__ float2 operator +=(float2 &a, float2 &b) {
    a.x = a.x - b.x;
    a.y = a.y - b.y;

    return a;
}

inline __host__ __device__ float2 operator *(const float2 &a, float scalar) {
    return {a.x * scalar, a.y * scalar};
}

inline __host__ __device__ float2 operator /(float2 &a, float scalar) {
    return {a.x / scalar, a.y / scalar};
}

__device__ float kernelDistanceAB(const Boid &boidSelf, const Boid &boidNeighbor);
__device__ float2 kernelCalculateAwayVector(Boid &boidSelf, Boid &boidNeighbor);
__device__ float2 kernelCalculateAcceleration(const float2 &separation, const float2 &alignment, const float2 &cohesion);
__global__ void kernelMakeAcceleration(Boid *boids, float2 *accelerationsOut);

__global__ void kernelInitBoids(Boid *boids, curandState *states, unsigned int seed);

#endif //CUDAPRACTICE_DEVICEFUNCTIONS_CUH