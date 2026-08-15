//
// Created by elder on 8/14/2026.
//

#ifndef CUDAPRACTICE_DEVICEFUNCTIONS_CUH
#define CUDAPRACTICE_DEVICEFUNCTIONS_CUH


struct Boid {
    bool valid = false;
    float2 position = {};
    float2 velocity = {};
};

inline float2 operator -(float2 &a, float2 &b) {
    return {a.x - b.x, a.y - b.y};
}

__device__ float kernelDistanceAB(const Boid &boidSelf, const Boid &boidNeighbor);
__device__ float2 kernelCalculateAwayVector(Boid &boidSelf, Boid &boidNeighbor);

#endif //CUDAPRACTICE_DEVICEFUNCTIONS_CUH