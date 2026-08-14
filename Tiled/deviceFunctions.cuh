//
// Created by elder on 8/12/2026.
//

#ifndef CUDAPRACTICE_DEVICEFUNCTIONS_CUH
#define CUDAPRACTICE_DEVICEFUNCTIONS_CUH

constexpr int N_BOIDS = 100;
constexpr int TPB = 64;
constexpr int BLOCKS = (N_BOIDS + TPB - 1) / TPB;
constexpr float DT = 0.016;
constexpr float MAX_TIME = 10.0f;
constexpr float SEPARATION_WEIGHT = 1.5f;
constexpr float ALIGNMENT_WEIGHT = 1.0f;
constexpr float COHESION_WEIGHT = 1.0f;
constexpr float PERCEPTION_RADIUS = 15.0f;
constexpr float SEPARATION_RADIUS = 5.0f;

struct Boid {
    bool valid = false;
    float2 position = {0.0f, 0.0f};
    float2 velocity = {0.0f, 0.0f};
};


inline __host__ __device__ float2 operator -(const float2 a, const float2 b) {
    return {a.x - b.x, a.y - b.y};
}

inline __host__ __device__ float2 operator +(const float2 &a, const float2 &b) {
    return {a.x + b.x, a.y + b.y};
}
inline __host__ __device__ float2 operator *(const float2 &a, const float scaler) {
    return {a.x * scaler, a.y * scaler};
}

inline __host__ __device__ float2 operator *(const float scaler, const float2 &a) {
    return a * scaler;
}

inline __host__ __device__ float2 operator +=(float2 &a, float2 &b) {
    return {a.x + b.x, a.y + b.y};
}

inline __host__ __device__ float2 operator /(float2 &a, int scalar) {
    return {a.x / static_cast<float>(scalar), a.y / static_cast<float>(scalar)};
}


__device__ float kernelDistanceBoidAB(const Boid &boidSelf, const Boid &boidNeighbor);
__device__ float2 kernelCalculateAwayVector(Boid &boidSelf, Boid &boidNeighbor);
__device__ float2 kernelSeparationAverage(const float2 *validAwayVectors, int validBoidCount);
__device__ float2 kernelAlignment(const Boid *boids, int validBoids);
__device__ float2 kernelCohesionAverage(const Boid *validBoids, int validBoidCount);
__device__ float2 kernelCalculateAcceleration(float2 averageSeparation, float2 averageAlignment, float2 averageCohesion);
__global__ void kernelMakeBoidAcceleration(Boid *boids, float2 *accelerationOut);
__global__ void integrateBoids(Boid *boids, float2 *accelerationIn);


#endif //CUDAPRACTICE_DEVICEFUNCTIONS_CUH