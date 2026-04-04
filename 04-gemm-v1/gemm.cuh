#pragma once

#include <cassert>
#include <cstddef>

#include <cuda_fp16.h>

#include <cuda_helpers.h>

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;  // Distance in elements between first values of consecutive rows/columns
    MatrixLayout layout;
};

__global__ void GEMMKernel(const DeviceMatrix a, const DeviceMatrix b, const DeviceMatrix c,
                           DeviceMatrix d, float alpha, float beta) {
    const size_t tileSize = 64;
    size_t localCol = threadIdx.x;
    size_t blockRow = blockIdx.y * tileSize;
    size_t blockCol = blockIdx.x * tileSize;

    __shared__ __half tileA[tileSize][tileSize];
    __shared__ __half tileB[tileSize][tileSize];

    float acc[tileSize];
    for (size_t row = 0; row < tileSize; ++row) {
        acc[row] = 0.0f;
    }

    for (size_t tileK = 0; tileK < a.cols; tileK += tileSize) {
        for (size_t row = 0; row < tileSize; ++row) {
            size_t aRow = blockRow + row;
            size_t aCol = tileK + localCol;
            if (aRow < a.rows && aCol < a.cols) {
                tileA[row][localCol] = a.data[aRow * a.stride + aCol];
            } else {
                tileA[row][localCol] = __float2half_rn(0.0f);
            }

            size_t bRow = tileK + row;
            size_t bCol = blockCol + localCol;
            if (bRow < b.rows && bCol < b.cols) {
                tileB[row][localCol] = b.data[bCol * b.stride + bRow];
            } else {
                tileB[row][localCol] = __float2half_rn(0.0f);
            }
        }

        __syncthreads();

        for (size_t k = 0; k < tileSize; ++k) {
            __half bValue = tileB[k][localCol];
            for (size_t row = 0; row < tileSize; ++row) {
                __half aValue = tileA[row][k];
                acc[row] += __half2float(__hmul(aValue, bValue));
            }
        }

        __syncthreads();
    }

    size_t globalCol = blockCol + localCol;
    if (globalCol >= d.cols) {
        return;
    }

    for (size_t row = 0; row < tileSize; ++row) {
        size_t globalRow = blockRow + row;
        if (globalRow >= d.rows) {
            continue;
        }

        size_t offset = globalCol * d.stride + globalRow;
        float cValue = __half2float(c.data[offset]);
        d.data[offset] = __float2half_rn(alpha * acc[row] + beta * cValue);
    }
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(a.layout == MatrixLayout::RowMajor);
    assert(b.layout == MatrixLayout::ColMajor);
    assert(c.layout == MatrixLayout::ColMajor);
    assert(d.layout == MatrixLayout::ColMajor);

    size_t tileSize = 64;
    dim3 block_size(tileSize);
    dim3 grid_size((d.cols + tileSize - 1) / tileSize, (d.rows + tileSize - 1) / tileSize);
    GEMMKernel<<<grid_size, block_size>>>(a, b, c, d, alpha, beta);
}
