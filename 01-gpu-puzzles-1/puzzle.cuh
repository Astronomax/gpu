#pragma once

__global__ void Map(const float* data, float* out) {
    out[threadIdx.x] = data[threadIdx.x] + 10;
}

__global__ void Zip(const float* left, const float* right, float* out) {
    out[threadIdx.x] = left[threadIdx.x] + right[threadIdx.x];
}

__global__ void Guard(const float* data, float* out, size_t size) {
    if (threadIdx.x < size) {
        out[threadIdx.x] = data[threadIdx.x] + 10;
    }
}

__global__ void Block(const float* data, float* out, float value, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = data[index] + value;
    }
}
