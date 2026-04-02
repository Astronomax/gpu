#include "quaternions.cuh"

#include <cassert>

#include <cuda_runtime.h>

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream) {
    // YOUR CODE HERE
    // NB: no need to do any allocations here
    // NB: no explicit cudaDeviceSynchronize is required here
}
