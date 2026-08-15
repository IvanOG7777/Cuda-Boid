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

