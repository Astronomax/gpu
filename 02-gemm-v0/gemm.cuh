#pragma once

#include <cuda_fp16.h>

#include <cstddef>

#include <cuda_helpers.h>

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;  // Distance in elements between first values of consecutive rows/columns
    MatrixLayout layout;
};

__device__ size_t GetMatrixOffset(const DeviceMatrix& matrix, size_t row, size_t col) {
    if (matrix.layout == MatrixLayout::RowMajor) {
        return row * matrix.stride + col;
    }
    return col * matrix.stride + row;
}

__global__ void GEMMKernel(const DeviceMatrix a, const DeviceMatrix b, const DeviceMatrix c,
                           DeviceMatrix d, float alpha, float beta) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d.rows || col >= d.cols) {
        return;
    }

    float accumulator = 0.0f;
    for (size_t inner = 0; inner < a.cols; ++inner) {
        __half a_value = a.data[GetMatrixOffset(a, row, inner)];
        __half b_value = b.data[GetMatrixOffset(b, inner, col)];
        __half product = __hmul(a_value, b_value);
        accumulator += __half2float(product);
    }

    float c_value = __half2float(c.data[GetMatrixOffset(c, row, col)]);
    d.data[GetMatrixOffset(d, row, col)] = __float2half_rn(alpha * accumulator + beta * c_value);
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    size_t kBlockSizeX = 16;
    size_t kBlockSizeY = 16;
    dim3 block_size(kBlockSizeX, kBlockSizeY);
    dim3 grid_size((d.cols + kBlockSizeX - 1) / kBlockSizeX,
                   (d.rows + kBlockSizeY - 1) / kBlockSizeY);

    GEMMKernel<<<grid_size, block_size>>>(a, b, c, d, alpha, beta);
}
