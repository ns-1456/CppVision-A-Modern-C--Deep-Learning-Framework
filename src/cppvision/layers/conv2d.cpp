#include "cppvision/layers/conv2d.h"
#include "cppvision/core/tensor.h"
#include <random>
#include <cmath>

namespace cppvision {

// Constructor with square kernel
Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, 
               int stride, int padding, bool use_bias, const std::string& name)
    : Conv2D(in_channels, out_channels, kernel_size, kernel_size,
             stride, stride, padding, padding, use_bias, name) {
    // Delegates to main constructor
}

// Main constructor
Conv2D::Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
               int stride_h, int stride_w, int padding_h, int padding_w,
               bool use_bias, const std::string& name)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_height_(kernel_h), kernel_width_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w),
      padding_h_(padding_h), padding_w_(padding_w),
      use_bias_(use_bias), name_(name) {
    // TODO: Implement Conv2D constructor
    // Algorithm:
    //   1. Initialize kernel tensor with shape [out_channels, in_channels, kernel_h, kernel_w]
    std::vector<int> kernel_shape = {out_channels_, in_channels_, kernel_height_, kernel_width_};
    kernels_ = Tensor::random_normal(kernel_shape, 0.0, std::sqrt(2.0 / (in_channels_ * kernel_height_ * kernel_width_)), true);
    
    //   2. Initialize bias tensor with shape [out_channels] if use_bias
    if (use_bias_) {
        std::vector<int> bias_shape = {out_channels_};
        biases_ = Tensor::zeros(bias_shape, true);
    }
    
    //   3. Initialize gradient tensors with same shapes
    kernel_gradients_ = Tensor::zeros(kernel_shape, false);
    if (use_bias_) {
        std::vector<int> bias_shape = {out_channels_};
        bias_gradients_ = Tensor::zeros(bias_shape, false);
    }
    
    //   4. Apply He initialization to kernels
    he_init();
    //   5. Initialize biases to zero
    // Hint: Use Tensor::random_normal() for kernel initialization
}

// Forward pass
Tensor Conv2D::forward(const Tensor& input) {
    // TODO: Implement Conv2D forward pass
    // Algorithm:
    //   1. Cache input for backward pass
    input_cache_ = input;
    
    // Extract dimensions
    std::vector<int> input_shape = input.shape();
    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];
    
    //   2. Add padding to input if needed
    Tensor padded_input = input;
    if (padding_h_ > 0 || padding_w_ > 0) {
        padded_input = add_padding(input);
    }
    
    //   3. Perform convolution using im2col transformation
    Tensor col = im2col(padded_input, batch_size, 
                       height + 2 * padding_h_, width + 2 * padding_w_);
    
    // Reshape kernels for matrix multiplication: [out_channels, in_channels * kernel_h * kernel_w]
    std::vector<int> kernel_2d_shape = {out_channels_, in_channels_ * kernel_height_ * kernel_width_};
    Tensor kernels_2d = kernels_.reshape(kernel_2d_shape);
    
    // Perform matrix multiplication for each batch
    int out_height = calculate_output_height(height);
    int out_width = calculate_output_width(width);
    std::vector<int> output_shape = {batch_size, out_channels_, out_height, out_width};
    Tensor output = Tensor::zeros(output_shape, input.requires_grad());
    
    for (int b = 0; b < batch_size; ++b) {
        // Extract batch from col: [in_channels * kernel_h * kernel_w, out_h * out_w]
        std::vector<int> col_batch_shape = {in_channels_ * kernel_height_ * kernel_width_, out_height * out_width};
        Tensor col_batch = Tensor::zeros(col_batch_shape, input.requires_grad());
        
        // Copy data for this batch
        for (int i = 0; i < col_batch.size(); ++i) {
            int col_idx = b * col_batch.size() + i;
            col_batch[i] = col[col_idx];
        }
        
        // Matrix multiplication: kernels_2d @ col_batch
        Tensor batch_output = kernels_2d.matmul(col_batch);
        
        // Reshape and store result
        std::vector<int> batch_output_shape = {out_channels_, out_height, out_width};
        batch_output = batch_output.reshape(batch_output_shape);
        
        // Copy to output tensor
        for (int i = 0; i < batch_output.size(); ++i) {
            int output_idx = b * batch_output.size() + i;
            output[output_idx] = batch_output[i];
        }
    }
    
    //   4. Apply bias if use_bias
    if (use_bias_) {
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels_; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        int output_idx = b * out_channels_ * out_height * out_width + 
                                       c * out_height * out_width + 
                                       h * out_width + w;
                        output[output_idx] += biases_[c];
                    }
                }
            }
        }
    }
    
    //   5. Cache output for backward pass
    output_cache_ = output;
    //   6. Return output tensor
    return output;
    // Hint: Use im2col() helper function for efficient convolution
    // Note: input shape should be [batch_size, in_channels, height, width]
    //       output shape will be [batch_size, out_channels, out_height, out_width]
}

// Backward pass
Tensor Conv2D::backward(const Tensor& grad_output) {
    // TODO: Implement Conv2D backward pass
    // Algorithm:
    //   1. Compute kernel gradients using convolution
    std::vector<int> input_shape = input_cache_.shape();
    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];
    
    // Add padding to input cache for gradient computation
    Tensor padded_input = input_cache_;
    if (padding_h_ > 0 || padding_w_ > 0) {
        padded_input = add_padding(input_cache_);
    }
    
    // Convert input to column format
    Tensor col = im2col(padded_input, batch_size, 
                       height + 2 * padding_h_, width + 2 * padding_w_);
    
    // Reshape grad_output for matrix multiplication
    int out_height = calculate_output_height(height);
    int out_width = calculate_output_width(width);
    std::vector<int> grad_2d_shape = {out_channels_, out_height * out_width};
    
    // Compute kernel gradients: grad_kernels = grad_output @ col^T
    for (int b = 0; b < batch_size; ++b) {
        // Extract batch from grad_output
        Tensor grad_batch = Tensor::zeros(grad_2d_shape, false);
        for (int i = 0; i < grad_batch.size(); ++i) {
            int grad_idx = b * grad_batch.size() + i;
            grad_batch[i] = grad_output[grad_idx];
        }
        
        // Extract batch from col
        std::vector<int> col_batch_shape = {in_channels_ * kernel_height_ * kernel_width_, out_height * out_width};
        Tensor col_batch = Tensor::zeros(col_batch_shape, false);
        for (int i = 0; i < col_batch.size(); ++i) {
            int col_idx = b * col_batch.size() + i;
            col_batch[i] = col[col_idx];
        }
        
        // Compute gradients: grad_kernels += grad_batch @ col_batch^T
        Tensor batch_kernel_grads = grad_batch.matmul(col_batch.transpose());
        
        // Add to accumulated kernel gradients
        for (int i = 0; i < kernels_.size(); ++i) {
            kernel_gradients_[i] += batch_kernel_grads[i];
        }
    }
    
    //   2. Compute bias gradients: sum over spatial dimensions if use_bias
    if (use_bias_) {
        for (int c = 0; c < out_channels_; ++c) {
            double bias_grad = 0.0;
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        int grad_idx = b * out_channels_ * out_height * out_width + 
                                     c * out_height * out_width + 
                                     h * out_width + w;
                        bias_grad += grad_output[grad_idx];
                    }
                }
            }
            bias_gradients_[c] = bias_grad;
        }
    }
    
    //   3. Compute input gradients using transposed convolution
    // Reshape kernels for transposed convolution
    std::vector<int> kernel_2d_shape = {out_channels_, in_channels_ * kernel_height_ * kernel_width_};
    Tensor kernels_2d = kernels_.reshape(kernel_2d_shape);
    
    // Compute input gradients
    std::vector<int> grad_input_shape = {batch_size, in_channels_, height, width};
    Tensor grad_input = Tensor::zeros(grad_input_shape, grad_output.requires_grad());
    
    for (int b = 0; b < batch_size; ++b) {
        // Extract batch from grad_output
        Tensor grad_batch = Tensor::zeros(grad_2d_shape, false);
        for (int i = 0; i < grad_batch.size(); ++i) {
            int grad_idx = b * grad_batch.size() + i;
            grad_batch[i] = grad_output[grad_idx];
        }
        
        // Transpose kernels for transposed convolution
        Tensor kernels_t = kernels_2d.transpose();
        
        // Matrix multiplication: kernels_t @ grad_batch
        Tensor batch_grad_input = kernels_t.matmul(grad_batch);
        
        // Reshape to column format
        std::vector<int> col_shape = {in_channels_ * kernel_height_ * kernel_width_, out_height * out_width};
        batch_grad_input = batch_grad_input.reshape(col_shape);
        
        // Convert back to image format
        Tensor batch_grad_image = col2im(batch_grad_input, 1, height + 2 * padding_h_, width + 2 * padding_w_);
        
        // Remove padding
        if (padding_h_ > 0 || padding_w_ > 0) {
            batch_grad_image = remove_padding(batch_grad_image, height, width);
        }
        
        // Copy to grad_input
        for (int i = 0; i < batch_grad_image.size(); ++i) {
            int grad_input_idx = b * batch_grad_image.size() + i;
            grad_input[grad_input_idx] = batch_grad_image[i];
        }
    }
    
    //   4. Store gradients in member variables (already done above)
    //   5. Return input gradients
    return grad_input;
    // Hint: Use col2im() helper function for gradient computation
}

// Parameter management
std::vector<Tensor> Conv2D::get_parameters() const {
    // TODO: Implement parameter retrieval
    // Algorithm:
    //   1. Create vector of parameters
    std::vector<Tensor> params;
    //   2. Add kernels tensor
    params.push_back(kernels_);
    //   3. Add biases tensor if use_bias is true
    if (use_bias_) {
        params.push_back(biases_);
    }
    //   4. Return parameter vector
    return params;
}

std::vector<Tensor> Conv2D::get_gradients() const {
    // TODO: Implement gradient retrieval
    // Algorithm:
    //   1. Create vector of gradients
    std::vector<Tensor> grads;
    //   2. Add kernel gradients tensor
    grads.push_back(kernel_gradients_);
    //   3. Add bias gradients tensor if use_bias is true
    if (use_bias_) {
        grads.push_back(bias_gradients_);
    }
    //   4. Return gradient vector
    return grads;
}

void Conv2D::update_parameters(const std::vector<Tensor>& gradients) {
    // TODO: Implement parameter update
    // Algorithm:
    //   1. Update kernels: kernels = kernels - gradients[0]
    for (int i = 0; i < kernels_.size(); ++i) {
        kernels_[i] -= gradients[0][i];
    }
    //   2. Update biases: biases = biases - gradients[1] if use_bias
    if (use_bias_ && gradients.size() > 1) {
        for (int i = 0; i < biases_.size(); ++i) {
            biases_[i] -= gradients[1][i];
        }
    }
    //   3. Handle gradient indexing properly
    // Hint: This method is called by optimizer, gradients are already scaled by learning rate
}

void Conv2D::zero_gradients() {
    // TODO: Implement gradient zeroing
    // Algorithm:
    //   1. Set all kernel gradients to zero
    for (int i = 0; i < kernel_gradients_.size(); ++i) {
        kernel_gradients_[i] = 0.0;
    }
    //   2. Set all bias gradients to zero if use_bias
    if (use_bias_) {
        for (int i = 0; i < bias_gradients_.size(); ++i) {
            bias_gradients_[i] = 0.0;
        }
    }
    // Hint: Use tensor operations to fill with zeros
}

// Layer information
int Conv2D::num_parameters() const {
    // TODO: Implement parameter counting
    // Algorithm:
    //   1. Calculate kernel parameters: out_channels * in_channels * kernel_h * kernel_w
    int kernel_params = out_channels_ * in_channels_ * kernel_height_ * kernel_width_;
    //   2. Add bias parameters: out_channels if use_bias
    int bias_params = use_bias_ ? out_channels_ : 0;
    //   3. Return total parameter count
    return kernel_params + bias_params;
}

// Weight initialization methods
void Conv2D::xavier_init() {
    // TODO: Implement Xavier initialization for Conv2D
    // Algorithm:
    //   1. Calculate fan_in and fan_out for convolution
    int fan_in = in_channels_ * kernel_height_ * kernel_width_;
    int fan_out = out_channels_ * kernel_height_ * kernel_width_;
    //   2. Calculate standard deviation: sqrt(2.0 / (fan_in + fan_out))
    double std_dev = std::sqrt(2.0 / (fan_in + fan_out));
    //   3. Generate random kernels from normal distribution
    std::vector<int> kernel_shape = {out_channels_, in_channels_, kernel_height_, kernel_width_};
    kernels_ = Tensor::random_normal(kernel_shape, 0.0, std_dev, true);
    //   4. Update kernels tensor
    // Hint: fan_in = in_channels * kernel_h * kernel_w
    //       fan_out = out_channels * kernel_h * kernel_w
}

void Conv2D::he_init() {
    // TODO: Implement He initialization for Conv2D
    // Algorithm:
    //   1. Calculate fan_in for convolution
    int fan_in = in_channels_ * kernel_height_ * kernel_width_;
    //   2. Calculate standard deviation: sqrt(2.0 / fan_in)
    double std_dev = std::sqrt(2.0 / fan_in);
    //   3. Generate random kernels from normal distribution
    std::vector<int> kernel_shape = {out_channels_, in_channels_, kernel_height_, kernel_width_};
    kernels_ = Tensor::random_normal(kernel_shape, 0.0, std_dev, true);
    //   4. Update kernels tensor
    // Hint: fan_in = in_channels * kernel_h * kernel_w
}

void Conv2D::uniform_init(double min, double max) {
    // TODO: Implement uniform initialization
    // Algorithm:
    //   1. Generate random kernels from uniform distribution
    std::vector<int> kernel_shape = {out_channels_, in_channels_, kernel_height_, kernel_width_};
    kernels_ = Tensor::random(kernel_shape, min, max, true);
    //   2. Update kernels tensor
    // Hint: Use Tensor::random() with min and max values
}

void Conv2D::normal_init(double mean, double std) {
    // TODO: Implement normal initialization
    // Algorithm:
    //   1. Generate random kernels from normal distribution
    std::vector<int> kernel_shape = {out_channels_, in_channels_, kernel_height_, kernel_width_};
    kernels_ = Tensor::random_normal(kernel_shape, mean, std, true);
    //   2. Update kernels tensor
    // Hint: Use Tensor::random_normal() with mean and std
}

// Utility methods
int Conv2D::calculate_output_height(int input_h) const {
    // TODO: Implement output height calculation
    // Algorithm:
    //   1. Apply convolution formula: (input_h + 2*padding_h - kernel_h) / stride_h + 1
    int output_h = (input_h + 2 * padding_h_ - kernel_height_) / stride_h_ + 1;
    //   2. Return calculated output height
    return output_h;
}

int Conv2D::calculate_output_width(int input_w) const {
    // TODO: Implement output width calculation
    // Algorithm:
    //   1. Apply convolution formula: (input_w + 2*padding_w - kernel_w) / stride_w + 1
    int output_w = (input_w + 2 * padding_w_ - kernel_width_) / stride_w_ + 1;
    //   2. Return calculated output width
    return output_w;
}

// Private helper methods
Tensor Conv2D::im2col(const Tensor& input, int batch_size, int height, int width) const {
    // TODO: Implement im2col transformation
    // Algorithm:
    //   1. Create output tensor for column format
    int out_height = calculate_output_height(height);
    int out_width = calculate_output_width(width);
    int col_height = in_channels_ * kernel_height_ * kernel_width_;
    int col_width = out_height * out_width;
    
    std::vector<int> col_shape = {batch_size, col_height, col_width};
    Tensor col = Tensor::zeros(col_shape, input.requires_grad());
    
    //   2. For each spatial location in output:
    for (int b = 0; b < batch_size; ++b) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
            for (int out_w = 0; out_w < out_width; ++out_w) {
                int col_idx = out_h * out_width + out_w;
                
                // Extract kernel-sized patch from input
                for (int c = 0; c < in_channels_; ++c) {
                    for (int k_h = 0; k_h < kernel_height_; ++k_h) {
                        for (int k_w = 0; k_w < kernel_width_; ++k_w) {
                            int in_h = out_h * stride_h_ + k_h - padding_h_;
                            int in_w = out_w * stride_w_ + k_w - padding_w_;
                            
                            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                                int input_idx = b * in_channels_ * height * width + 
                                              c * height * width + 
                                              in_h * width + in_w;
                                int col_element_idx = c * kernel_height_ * kernel_width_ + 
                                                    k_h * kernel_width_ + k_w;
                                int col_tensor_idx = b * col_height * col_width + 
                                                   col_element_idx * col_width + col_idx;
                                col[col_tensor_idx] = input[input_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    
    //   3. Return transformed tensor
    return col;
    // Hint: This enables efficient convolution using matrix multiplication
}

Tensor Conv2D::col2im(const Tensor& grad_col, int batch_size, int height, int width) const {
    // TODO: Implement col2im transformation
    // Algorithm:
    //   1. Create output tensor for image format
    std::vector<int> output_shape = {batch_size, in_channels_, height, width};
    Tensor output = Tensor::zeros(output_shape, grad_col.requires_grad());
    
    int out_height = calculate_output_height(height);
    int out_width = calculate_output_width(width);
    int col_height = in_channels_ * kernel_height_ * kernel_width_;
    int col_width = out_height * out_width;
    
    //   2. For each spatial location:
    for (int b = 0; b < batch_size; ++b) {
        for (int out_h = 0; out_h < out_height; ++out_h) {
            for (int out_w = 0; out_w < out_width; ++out_w) {
                int col_idx = out_h * out_width + out_w;
                
                // Extract column from grad_col and reshape to kernel size
                for (int c = 0; c < in_channels_; ++c) {
                    for (int k_h = 0; k_h < kernel_height_; ++k_h) {
                        for (int k_w = 0; k_w < kernel_width_; ++k_w) {
                            int in_h = out_h * stride_h_ + k_h - padding_h_;
                            int in_w = out_w * stride_w_ + k_w - padding_w_;
                            
                            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                                int col_element_idx = c * kernel_height_ * kernel_width_ + 
                                                    k_h * kernel_width_ + k_w;
                                int col_tensor_idx = b * col_height * col_width + 
                                                   col_element_idx * col_width + col_idx;
                                int output_idx = b * in_channels_ * height * width + 
                                              c * height * width + 
                                              in_h * width + in_w;
                                output[output_idx] += grad_col[col_tensor_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    
    //   3. Return transformed gradient tensor
    return output;
    // Hint: This is used for computing input gradients
}

Tensor Conv2D::add_padding(const Tensor& input) const {
    // TODO: Implement padding addition
    // Algorithm:
    //   1. Create output tensor with padded dimensions
    std::vector<int> input_shape = input.shape();
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    
    int padded_height = height + 2 * padding_h_;
    int padded_width = width + 2 * padding_w_;
    std::vector<int> padded_shape = {batch_size, channels, padded_height, padded_width};
    
    Tensor padded = Tensor::zeros(padded_shape, input.requires_grad());
    
    //   2. Copy input data to center of padded tensor
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int input_idx = b * channels * height * width + c * height * width + h * width + w;
                    int padded_idx = b * channels * padded_height * padded_width + 
                                   c * padded_height * padded_width + 
                                   (h + padding_h_) * padded_width + (w + padding_w_);
                    padded[padded_idx] = input[input_idx];
                }
            }
        }
    }
    
    //   3. Fill padding regions with zeros (already done by zeros initialization)
    //   4. Return padded tensor
    return padded;
    // Hint: Handle different padding modes (zero, reflect, replicate)
}

Tensor Conv2D::remove_padding(const Tensor& grad_padded, int original_height, int original_width) const {
    // TODO: Implement padding removal
    // Algorithm:
    //   1. Extract center region from padded gradient tensor
    std::vector<int> padded_shape = grad_padded.shape();
    int batch_size = padded_shape[0];
    int channels = padded_shape[1];
    
    std::vector<int> output_shape = {batch_size, channels, original_height, original_width};
    Tensor output = Tensor::zeros(output_shape, grad_padded.requires_grad());
    
    //   2. Return unpadded gradient tensor
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < original_height; ++h) {
                for (int w = 0; w < original_width; ++w) {
                    int padded_idx = b * channels * padded_shape[2] * padded_shape[3] + 
                                   c * padded_shape[2] * padded_shape[3] + 
                                   (h + padding_h_) * padded_shape[3] + (w + padding_w_);
                    int output_idx = b * channels * original_height * original_width + 
                                    c * original_height * original_width + 
                                    h * original_width + w;
                    output[output_idx] = grad_padded[padded_idx];
                }
            }
        }
    }
    
    return output;
    // Hint: Remove padding_h_ rows from top/bottom and padding_w_ columns from left/right
}

} // namespace cppvision
