//
// Created by elder on 8/12/2026.
//

#include <iostream>
#include "deviceFunctions.cuh"

int main() {
    Boid *deviceBoids = nullptr;
    cudaError_t boidsMalloc = cudaMalloc(&deviceBoids, N_BOIDS * sizeof(Boid));
    if (boidsMalloc != cudaSuccess) {
        std:: cerr << "Fail to allocate memory for device boids\n";
        exit(EXIT_FAILURE);
    }
    return 0;
}