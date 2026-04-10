#include "quaternions.cuh"

#include <cassert>
#include <cuda_runtime.h>

__device__ void WarpReduceQuaternions(Quaternion* shared, size_t tid) {
    #pragma unroll
    for (size_t shift = 1; shift < 32; shift <<= 1) {
        if (tid % (2 * shift) == 0) {
            shared[tid] = QuaternionMultiplier()(shared[tid], shared[tid + shift]);
        }
        __syncwarp();
    }
}

__global__ void QuaternionsReduceKernel(size_t rows, size_t cols,
                                        const Quaternion* d_input_matrix, size_t inp_stride,
                                        Quaternion* d_out) {
    size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const size_t blockSize = 256;
    __shared__ Quaternion shared_quaternions[blockSize];

    size_t tid = threadIdx.x;
    assert(blockSize == blockDim.x);
    assert(blockDim.x == blockSize);
    assert(cols % blockSize == 0);

    size_t items_per_thread = cols / blockDim.x;
    assert(items_per_thread == 4 || items_per_thread == 8 || items_per_thread == 16);

    size_t batch_begin = tid * items_per_thread;
    size_t batch_end = batch_begin + items_per_thread;
    assert(batch_end <= cols);

    const Quaternion* input_row = d_input_matrix + row * inp_stride;
    Quaternion local{1.0f, 0.0f, 0.0f, 0.0f};

    if (batch_begin < cols) {
        local = input_row[batch_begin];
        for (size_t col = batch_begin + 1; col < batch_end; ++col) {
            local = QuaternionMultiplier()(local, input_row[col]);
        }
    }
    shared_quaternions[tid] = local;

    __syncwarp();
    WarpReduceQuaternions(shared_quaternions, tid);

    __syncthreads();
    #pragma unroll
    for (size_t shift = 32; shift < blockSize; shift <<= 1) {
        if (tid % (2 * shift) == 0) {
            shared_quaternions[tid] = QuaternionMultiplier()(shared_quaternions[tid], shared_quaternions[tid + shift]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[row] = shared_quaternions[0];
    }
}

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream) {
    QuaternionsReduceKernel<<<rows, 256, 0, stream>>>(rows, cols, inp, inp_stride, out);
}
