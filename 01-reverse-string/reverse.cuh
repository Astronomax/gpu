#pragma once

#include <cstddef>

#include <cuda_helpers.h>

__global__ void ReverseDeviceStringInplaceKernel(char* str, size_t length) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = start; idx < length / 2; idx += stride) {
        size_t idx_ = length - idx - 1;
        char tmp = str[idx];
        str[idx] = str[idx_];
        str[idx_] = tmp;
    }
}

void ReverseDeviceStringInplace(char* str, size_t length) {
    if (length < 2) {
        return;
    }

    size_t blockSize = 256;
    size_t maxBlocks = 65535;
    const size_t num_blocks = std::min(maxBlocks, (length / 2 + blockSize - 1) / blockSize);

    ReverseDeviceStringInplaceKernel<<<num_blocks, blockSize>>>(str, length);
    CheckStatus(cudaGetLastError());
    CheckStatus(cudaDeviceSynchronize());
    CheckStatus(cudaGetLastError());
}
