#include <cuda_fp16.h>

#include <cstddef>

#include <cuda_helpers.h>

__global__ void TransposeKernel(const __half* input_device, size_t input_stride,
                                __half* output_device, size_t output_stride, size_t rows,
                                size_t cols) {
    const size_t tileSize = 32;
    __shared__ __half tile[tileSize][tileSize + 1];

    size_t j = blockIdx.x * tileSize + threadIdx.x;
    size_t i = blockIdx.y * tileSize + threadIdx.y;

    if (i < rows && j < cols) {
        tile[threadIdx.y][threadIdx.x] = input_device[i * input_stride + j];
    }

    __syncthreads();

    i = blockIdx.y * tileSize + threadIdx.x;
    j = blockIdx.x * tileSize + threadIdx.y;

    if (i < rows && j < cols) {
        output_device[j * output_stride + i] = tile[threadIdx.x][threadIdx.y];
    }
}

void TransposeDevice(const __half* input_device, size_t input_stride, __half* output_device,
                     size_t output_stride, size_t rows, size_t cols) {
    size_t tileSize = 32;
    dim3 block_size(tileSize, tileSize);
    dim3 grid_size((cols + tileSize - 1) / tileSize, (rows + tileSize - 1) / tileSize);

    TransposeKernel<<<grid_size, block_size>>>(input_device, input_stride, output_device,
                                               output_stride, rows, cols);
}
