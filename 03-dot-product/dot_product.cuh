#pragma once

#include <cuda_helpers.h>

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    // YOUR CODE HERE
}

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    // YOUR CODE HERE
}
