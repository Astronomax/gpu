#pragma once

#include <cassert>

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

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(a.layout == MatrixLayout::RowMajor);
    assert(b.layout == MatrixLayout::ColMajor);
    assert(c.layout == MatrixLayout::ColMajor);
    assert(d.layout == MatrixLayout::ColMajor);
    // YOUR CODE HERE
}
