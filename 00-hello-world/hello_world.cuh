#pragma once

#include <cstdio>
#include <stdexcept>

#include <cuda_helpers.h>

__global__ void hello_world_kernel() {
    printf("Hello, world!");
}

void CallHelloWorld() {
    hello_world_kernel<<<1, 1>>>();
    CheckStatus(cudaDeviceSynchronize());
}
