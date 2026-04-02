#pragma once

#include <cstddef>
#include <vector>

#include <cuda_helpers.h>

__global__ void AddDeviceVectorsKernel(const float* left_device, const float* right_device,
                                       float* out_device, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out_device[idx] = left_device[idx] + right_device[idx];
    }
}

float* AllocDeviceVector(size_t size) {
    float* vector_device = nullptr;
    CheckStatus(cudaMalloc(&vector_device, size * sizeof(float)));
    return vector_device;
}

void FreeDeviceVector(float* vector_device) {
    CheckStatus(cudaFree(vector_device));
}

void CopyHostVectorToDevice(const std::vector<float>& src_host, float* dst_device) {
    CheckStatus(cudaMemcpy(dst_device, src_host.data(), src_host.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
}

std::vector<float> CopyDeviceVectorToHost(const float* src_device, size_t size) {
    std::vector<float> dst_host(size);
    CheckStatus(
        cudaMemcpy(dst_host.data(), src_device, size * sizeof(float), cudaMemcpyDeviceToHost));
    return dst_host;
}

void AddDeviceVectors(const float* left_device, const float* right_device, float* out_device,
                      size_t size) {
    constexpr size_t blockSize = 256;
    const size_t blocks = (size + blockSize - 1) / blockSize;
    AddDeviceVectorsKernel<<<blocks, blockSize>>>(left_device, right_device, out_device, size);
    CheckStatus(cudaGetLastError());
    CheckStatus(cudaDeviceSynchronize());
}
