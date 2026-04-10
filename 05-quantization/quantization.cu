#include "quantization.cuh"

#include <cmath>

#include <cuda_helpers.h>

__device__ float WarpReduceMax(float* shared, size_t lane) {
    #pragma unroll
    for (size_t shift = 16; shift > 0; shift >>= 1) {
        if (lane < shift) {
            shared[lane] = fmaxf(shared[lane], shared[lane + shift]);
        }
        __syncwarp();
    }
    return shared[0];
}

__global__ void QuantizationKernel(size_t rows, size_t cols, const float* d_input_matrix,
                                   const float* d_balance_factors, size_t input_stride,
                                   size_t out_stride, int8_t* d_out, float* d_out_scales) {
    size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float shared_max[256];

    size_t tid = threadIdx.x;
    float local_max = 1e-5f;
    const float* input_row = d_input_matrix + row * input_stride;
    int8_t* output_row = d_out + row * out_stride;

    for (size_t col = tid * 4; col < cols; col += blockDim.x * 4) {
        float4 input_vec = reinterpret_cast<const float4*>(input_row + col)[0];
        float4 balance_vec = reinterpret_cast<const float4*>(d_balance_factors + col)[0];

        float values[4] = {
            fabsf(input_vec.x + balance_vec.x),
            fabsf(input_vec.y + balance_vec.y),
            fabsf(input_vec.z + balance_vec.z),
            fabsf(input_vec.w + balance_vec.w),
        };

        #pragma unroll
        for (size_t i = 0; i < 4; ++i) {
            local_max = fmaxf(local_max, values[i]);
        }
    }

    shared_max[tid] = local_max;

    __syncthreads();
    for (size_t shift = blockDim.x / 2; shift >= 32; shift >>= 1) {
        if (tid < shift) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + shift]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceMax(shared_max, tid);
    }

    __syncthreads();
    float max_value = shared_max[0];
    float scale = 127.0f / max_value;

    if (tid == 0) {
        d_out_scales[row] = scale;
    }
    __syncthreads();

    for (size_t col = tid * 4; col < cols; col += blockDim.x * 4) {
        float4 input_vec = reinterpret_cast<const float4*>(input_row + col)[0];
        float4 balance_vec = reinterpret_cast<const float4*>(d_balance_factors + col)[0];

        output_row[col] = static_cast<int8_t>(roundf((input_vec.x + balance_vec.x) * scale));
        output_row[col + 1] = static_cast<int8_t>(roundf((input_vec.y + balance_vec.y) * scale));
        output_row[col + 2] = static_cast<int8_t>(roundf((input_vec.z + balance_vec.z) * scale));
        output_row[col + 3] = static_cast<int8_t>(roundf((input_vec.w + balance_vec.w) * scale));
    }
}

void Quantization(size_t rows, size_t cols, const float* d_input_matrix,
                  const float* d_balance_factors, size_t input_stride, size_t out_stride,
                  int8_t* d_out, float* d_out_scales) {
    size_t block_size = 256;
    QuantizationKernel<<<rows, block_size>>>(rows, cols, d_input_matrix, d_balance_factors,
                                                  input_stride, out_stride, d_out, d_out_scales);
}
