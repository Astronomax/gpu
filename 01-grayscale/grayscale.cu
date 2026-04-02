#include "grayscale.cuh"

#include <cstdlib>

#include <cuda_helpers.h>

__global__ void ConvertToGrayscaleKernel(const Image rgb_device_image, Image gray_device_image) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= rgb_device_image.width || y >= rgb_device_image.height) {
        return;
    }

    const uint8_t* src =
        rgb_device_image.pixels + y * rgb_device_image.stride + x * rgb_device_image.channels;
    uint8_t* dst = gray_device_image.pixels + y * gray_device_image.stride + x;

    float r = static_cast<float>(src[0]);
    float g = static_cast<float>(src[1]);
    float b = static_cast<float>(src[2]);
    *dst = static_cast<uint8_t>(r * 0.299f + g * 0.587f + b * 0.114f);
}

Image AllocHostImage(size_t width, size_t height, size_t channels) {
    size_t stride = width * channels;
    uint8_t* pixels = static_cast<uint8_t*>(std::malloc(height * stride));
    return Image{.pixels = pixels,
                 .width = width,
                 .height = height,
                 .stride = stride,
                 .channels = channels};
}

Image AllocDeviceImage(size_t width, size_t height, size_t channels) {
    Image image{.pixels = nullptr, .width = width, .height = height, .stride = 0, .channels = channels};
    size_t pitch = 0;
    CheckStatus(cudaMallocPitch(&image.pixels, &pitch, width * channels, height));
    image.stride = pitch;
    return image;
}

void CopyImageHostToDevice(const Image& src_host, Image& dst_device) {
    CheckStatus(cudaMemcpy2D(dst_device.pixels, dst_device.stride, src_host.pixels, src_host.stride,
                             src_host.width * src_host.channels, src_host.height,
                             cudaMemcpyHostToDevice));
}

void CopyImageDeviceToHost(const Image& src_device, Image& dst_host) {
    CheckStatus(cudaMemcpy2D(dst_host.pixels, dst_host.stride, src_device.pixels, src_device.stride,
                             src_device.width * src_device.channels, src_device.height,
                             cudaMemcpyDeviceToHost));
}

void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image) {
    size_t kBlockSizeX = 16;
    size_t kBlockSizeY = 16;
    dim3 block_size(kBlockSizeX, kBlockSizeY);
    dim3 grid_size((rgb_device_image.width + kBlockSizeX - 1) / kBlockSizeX,
                   (rgb_device_image.height + kBlockSizeY - 1) / kBlockSizeY);

    ConvertToGrayscaleKernel<<<grid_size, block_size>>>(rgb_device_image, gray_device_image);
    CheckStatus(cudaGetLastError());
}

void FreeDeviceImage(const Image& image) {
    CheckStatus(cudaFree(image.pixels));
}

void FreeHostImage(const Image& image) {
    std::free(image.pixels);
}
