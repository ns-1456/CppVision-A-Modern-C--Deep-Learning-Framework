#include "cppvision/layers/maxpool2d.h"
#include "cppvision/core/tensor.h"
#include <algorithm>
#include <limits>

namespace cppvision {

// Constructor with square kernel
MaxPool2D::MaxPool2D(int kernel_size, int stride, int padding, const std::string& name)
    : MaxPool2D(kernel_size, kernel_size, stride, stride, padding, padding, name) {
    // Delegates to main constructor
}

// Main constructor
MaxPool2D::MaxPool2D(int kernel_h, int kernel_w, int stride_h, int stride_w,
                     int padding_h, int padding_w, const std::string& name)
    : pool_height_(kernel_h), pool_width_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w), name_(name) {
    // Constructor - no parameters needed for pooling layer
}

// Forward pass
Tensor MaxPool2D::forward(const Tensor& input) {
    // MaxPool2D forward pass implementation
    // Algorithm:
    //   1. Cache input for backward pass
    input_cache_ = input;
    
    // Extract dimensions
    std::vector<int> input_shape = input.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    
    //   2. Calculate output dimensions
    int out_height = calculate_output_height(height);
    int out_width = calculate_output_width(width);
    
    //   3. Create output tensor
    std::vector<int> output_shape = {batch_size, channels, out_height, out_width};
    Tensor output = Tensor::zeros(output_shape, input.requires_grad());
    
    //   4. Perform max pooling for each sample, channel, and spatial location
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    // Find maximum value in kernel window
                    double max_val = std::numeric_limits<double>::lowest();
                    
                    for (int k_h = 0; k_h < pool_height_; ++k_h) {
                        for (int k_w = 0; k_w < pool_width_; ++k_w) {
                            int in_h = out_h * stride_h_ + k_h - padding_h_;
                            int in_w = out_w * stride_w_ + k_w - padding_w_;
                            
                            // Check bounds
                            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                                int input_idx = b * channels * height * width + 
                                              c * height * width + 
                                              in_h * width + in_w;
                                
                                max_val = std::max(max_val, input[input_idx]);
                            }
                        }
                    }
                    
                    // Store maximum value
                    int output_idx = b * channels * out_height * out_width + 
                                   c * out_height * out_width + 
                                   out_h * out_width + out_w;
                    output[output_idx] = max_val;
                }
            }
        }
    }
    
    //   5. Return output tensor
    return output;
}

// Backward pass
Tensor MaxPool2D::backward(const Tensor& grad_output) {
    // MaxPool2D backward pass implementation
    // Algorithm:
    //   1. Create input gradient tensor with same shape as input
    std::vector<int> input_shape = input_cache_.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    
    std::vector<int> grad_input_shape = {batch_size, channels, height, width};
    Tensor grad_input = Tensor::zeros(grad_input_shape, grad_output.requires_grad());
    
    //   2. Distribute gradients only to locations that contributed to max pooling
    int out_height = calculate_output_height(height);
    int out_width = calculate_output_width(width);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    int output_idx = b * channels * out_height * out_width + 
                                   c * out_height * out_width + 
                                   out_h * out_width + out_w;
                    
                    // Find the maximum element in the pooling window
                    double max_val = std::numeric_limits<double>::lowest();
                    int max_h = -1, max_w = -1;
                    
                    for (int k_h = 0; k_h < pool_height_; ++k_h) {
                        for (int k_w = 0; k_w < pool_width_; ++k_w) {
                            int in_h = out_h * stride_h_ + k_h - padding_h_;
                            int in_w = out_w * stride_w_ + k_w - padding_w_;
                            
                            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                                int input_idx = b * channels * height * width + 
                                              c * height * width + 
                                              in_h * width + in_w;
                                
                                if (input_cache_[input_idx] > max_val) {
                                    max_val = input_cache_[input_idx];
                                    max_h = in_h;
                                    max_w = in_w;
                                }
                            }
                        }
                    }
                    
                    // Gradient flows only to the maximum element
                    if (max_h >= 0 && max_w >= 0) {
                        int input_idx = b * channels * height * width + 
                                      c * height * width + 
                                      max_h * width + max_w;
                        grad_input[input_idx] += grad_output[output_idx];
                    }
                }
            }
        }
    }
    
    //   3. Return input gradients
    return grad_input;
}

// Utility methods
int MaxPool2D::calculate_output_height(int input_h) const {
    // Output height calculation implementation
    // Algorithm:
    //   1. Apply pooling formula: (input_h + 2*padding_h - pool_h) / stride_h + 1
    int output_h = (input_h + 2 * padding_h_ - pool_height_) / stride_h_ + 1;
    //   2. Return calculated output height
    return output_h;
}

int MaxPool2D::calculate_output_width(int input_w) const {
    // Output width calculation implementation
    // Algorithm:
    //   1. Apply pooling formula: (input_w + 2*padding_w - pool_w) / stride_w + 1
    int output_w = (input_w + 2 * padding_w_ - pool_width_) / stride_w_ + 1;
    //   2. Return calculated output width
    return output_w;
}

} // namespace cppvision
