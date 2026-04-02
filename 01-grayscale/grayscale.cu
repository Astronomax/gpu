#include "grayscale.cuh"

Image AllocHostImage(size_t width, size_t height, size_t channels) {
    // YOUR CODE HERE
}

Image AllocDeviceImage(size_t width, size_t height, size_t channels) {
    // YOUR CODE HERE
}

void CopyImageHostToDevice(const Image& src_host, Image& dst_device) {
    // YOUR CODE HERE
}

void CopyImageDeviceToHost(const Image& src_device, Image& dst_host) {
    // YOUR CODE HERE
}

void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image) {
    // YOUR CODE HERE
    // NB: output gray_device_image is expected to be preallocated, no need to do any allocations here
    // NB: no explicit cudaDeviceSynchronize is required here
}

void FreeDeviceImage(const Image& image) {
    // YOUR CODE HERE
}

void FreeHostImage(const Image& image) {
    // YOUR CODE HERE
}
