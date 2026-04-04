#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <cuda_helpers.h>

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    size_t blockSize = 256;
    size_t itemsPerThread = 4;
    size_t workspaceNumElements = (num_elements + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
    return workspaceNumElements * sizeof(float);
}

__device__ void ReduceBlockSum(float* shared, size_t threadId, size_t blockSize) {
    for (size_t stride = blockSize / 2; stride > 0; stride /= 2) {
        if (threadId < stride)
            shared[threadId] += shared[threadId + stride];
        __syncthreads();
    }
}

__global__ void DotProductBlockReduceKernel(const float* lhs_device, const float* rhs_device,
                                            size_t num_elements, float* workspace_device) {
    __shared__ float shared[256];

    size_t threadId = threadIdx.x;
    size_t blockSize = blockDim.x;
    size_t itemsPerThread = 4;
    size_t blockStart = blockIdx.x * blockSize * itemsPerThread;
    size_t threadStart = blockStart + threadId * itemsPerThread;

    float sum = 0.0f;
    if (threadStart + itemsPerThread <= num_elements) {
        float4 lhsVec = reinterpret_cast<const float4*>(lhs_device + threadStart)[0];
        float4 rhsVec = reinterpret_cast<const float4*>(rhs_device + threadStart)[0];
        sum += lhsVec.x * rhsVec.x;
        sum += lhsVec.y * rhsVec.y;
        sum += lhsVec.z * rhsVec.z;
        sum += lhsVec.w * rhsVec.w;
    } else {
        for (size_t idx = threadStart; idx < num_elements && idx < threadStart + itemsPerThread; ++idx) {
            sum += lhs_device[idx] * rhs_device[idx];
        }
    }

    shared[threadId] = sum;
    __syncthreads();

    ReduceBlockSum(shared, threadId, blockSize);

    if (threadId == 0)
        workspace_device[blockIdx.x] = shared[0];
}

__global__ void DotProductFinalReduceKernel(const float* workspace_device, size_t num_elements,
                                            float* out_device) {
    __shared__ float shared[256];

    size_t threadId = threadIdx.x;
    size_t blockSize = blockDim.x;
    float sum = 0.0f;
    for (size_t idx = threadId; idx < num_elements; idx += blockSize)
        sum += workspace_device[idx];

    shared[threadId] = sum;
    __syncthreads();

    ReduceBlockSum(shared, threadId, blockSize);

    if (threadId == 0)
        out_device[0] = shared[0];
}

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    if (num_elements == 0) {
        CheckStatus(cudaMemset(out_device, 0, sizeof(float)));
        return;
    }

    size_t blockSize = 256;
    size_t itemsPerThread = 4;
    size_t numBlocks = (num_elements + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);

    DotProductBlockReduceKernel<<<numBlocks, blockSize>>>(lhs_device, rhs_device, num_elements,
                                                          workspace_device);
    DotProductFinalReduceKernel<<<1, blockSize>>>(workspace_device, numBlocks, out_device);
}
