#pragma once

#include <vector>

#include <cuda_helpers.h>

float* AllocDeviceVector(size_t num_elements) {
    // YOUR CODE HERE
}

void FreeDeviceVector(float* device_ptr) {
    // YOUR CODE HERE
}

void CopyHostVectorToDevice(const std::vector<float>& vector_host, float* dst_device_ptr) {
    // YOUR CODE HERE
}

std::vector<float> CopyDeviceVectorToHost(const float* ptr_device, size_t num_elements) {
    // YOUR CODE HERE
}

void AddDeviceVectors(const float* left_device, const float* right_device, float* out_device,
                      size_t num_elements) {
    // YOUR CODE HERE
}
