#include "cppvision/layers/dense.h"
#include "cppvision/core/tensor.h"
#include <random>
#include <cmath>

namespace cppvision {

// Constructor
Dense::Dense(int input_size, int output_size, bool use_bias, const std::string& name)
    : input_size_(input_size), output_size_(output_size), use_bias_(use_bias), name_(name) {
    // Dense layer constructor implementation
    // Algorithm:
    //   1. Initialize weight matrix with shape [output_size, input_size]
    std::vector<int> weight_shape = {output_size_, input_size_};
    weights_ = Tensor::random_normal(weight_shape, 0.0, std::sqrt(2.0 / input_size_), true);
    
    //   2. Initialize bias vector with shape [output_size] if use_bias is true
    if (use_bias_) {
        std::vector<int> bias_shape = {output_size_};
        biases_ = Tensor::zeros(bias_shape, true);
    }
    
    //   3. Initialize gradient tensors with same shapes
    weight_gradients_ = Tensor::zeros(weight_shape, false);
    if (use_bias_) {
        std::vector<int> bias_shape = {output_size_};
        bias_gradients_ = Tensor::zeros(bias_shape, false);
    }
    
    //   4. Apply Xavier/Glorot initialization to weights
    xavier_init(input_size_, output_size_);
    //   5. Initialize biases to zero
    // Hint: Use Tensor::random_normal() for weight initialization
}

// Forward pass
Tensor Dense::forward(const Tensor& input) {
    // Dense layer forward pass implementation
    // Algorithm:
    //   1. Cache input for backward pass
    input_cache_ = input;
    
    // Extract dimensions
    std::vector<int> input_shape = input.shape();
    int batch_size = input_shape[0];
    
    //   2. Compute Y = X * W^T + b (matrix multiplication)
    // Reshape input to 2D: [batch_size, input_size]
    std::vector<int> input_2d_shape = {batch_size, input_size_};
    Tensor input_2d = input.reshape(input_2d_shape);
    
    // Matrix multiplication: Y = X * W^T
    Tensor output = input_2d.matmul(weights_.transpose());
    
    //   3. Add bias if use_bias
    if (use_bias_) {
        // Add bias to each sample in the batch
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < output_size_; ++i) {
                int output_idx = b * output_size_ + i;
                output[output_idx] += biases_[i];
            }
        }
    }
    
    //   4. Return output tensor
    return output;
    // Hint: Use tensor.matmul() for matrix multiplication
    // Note: input shape should be [batch_size, input_size]
    //       output shape will be [batch_size, output_size]
}

// Backward pass
Tensor Dense::backward(const Tensor& grad_output) {
    // TODO: Implement backward pass
    // Algorithm:
    //   1. Compute weight gradients: dW = grad_output^T * input
    std::vector<int> input_shape = input_cache_.shape();
    int batch_size = input_shape[0];
    
    // Reshape input to 2D: [batch_size, input_size]
    std::vector<int> input_2d_shape = {batch_size, input_size_};
    Tensor input_2d = input_cache_.reshape(input_2d_shape);
    
    // Reshape grad_output to 2D: [batch_size, output_size]
    std::vector<int> grad_2d_shape = {batch_size, output_size_};
    Tensor grad_2d = grad_output.reshape(grad_2d_shape);
    
    // Weight gradients: dW = grad_output^T * input
    Tensor weight_grads = grad_2d.transpose().matmul(input_2d);
    
    // Accumulate gradients (for batch processing)
    for (int i = 0; i < weights_.size(); ++i) {
        weight_gradients_[i] += weight_grads[i];
    }
    
    //   2. Compute bias gradients: db = sum(grad_output, axis=0) if use_bias
    if (use_bias_) {
        for (int i = 0; i < output_size_; ++i) {
            double bias_grad = 0.0;
            for (int b = 0; b < batch_size; ++b) {
                int grad_idx = b * output_size_ + i;
                bias_grad += grad_output[grad_idx];
            }
            bias_gradients_[i] += bias_grad;
        }
    }
    
    //   3. Compute input gradients: dX = grad_output * W
    Tensor input_grads = grad_2d.matmul(weights_);
    
    //   4. Store gradients in member variables (already done above)
    //   5. Return input gradients
    return input_grads;
    // Hint: Use tensor.matmul() and tensor operations
}

// Parameter management
std::vector<Tensor> Dense::get_parameters() const {
    // TODO: Implement parameter retrieval
    // Algorithm:
    //   1. Create vector of parameters
    std::vector<Tensor> params;
    //   2. Add weights tensor
    params.push_back(weights_);
    //   3. Add biases tensor if use_bias is true
    if (use_bias_) {
        params.push_back(biases_);
    }
    //   4. Return parameter vector
    return params;
}

std::vector<Tensor> Dense::get_gradients() const {
    // TODO: Implement gradient retrieval
    // Algorithm:
    //   1. Create vector of gradients
    std::vector<Tensor> grads;
    //   2. Add weight gradients tensor
    grads.push_back(weight_gradients_);
    //   3. Add bias gradients tensor if use_bias is true
    if (use_bias_) {
        grads.push_back(bias_gradients_);
    }
    //   4. Return gradient vector
    return grads;
}

void Dense::update_parameters(const std::vector<Tensor>& gradients) {
    // TODO: Implement parameter update
    // Algorithm:
    //   1. Update weights: weights = weights - gradients[0]
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] -= gradients[0][i];
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

void Dense::zero_gradients() {
    // TODO: Implement gradient zeroing
    // Algorithm:
    //   1. Set all weight gradients to zero
    for (int i = 0; i < weight_gradients_.size(); ++i) {
        weight_gradients_[i] = 0.0;
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
int Dense::num_parameters() const {
    // TODO: Implement parameter counting
    // Algorithm:
    //   1. Calculate weight parameters: output_size * input_size
    int weight_params = output_size_ * input_size_;
    //   2. Add bias parameters: output_size if use_bias
    int bias_params = use_bias_ ? output_size_ : 0;
    //   3. Return total parameter count
    return weight_params + bias_params;
}

// Weight initialization methods
void Dense::xavier_init(int fan_in, int fan_out) {
    // TODO: Implement Xavier/Glorot initialization
    // Algorithm:
    //   1. Calculate standard deviation: sqrt(2.0 / (fan_in + fan_out))
    double std_dev = std::sqrt(2.0 / (fan_in + fan_out));
    //   2. Generate random weights from normal distribution
    std::vector<int> weight_shape = {output_size_, input_size_};
    weights_ = Tensor::random_normal(weight_shape, 0.0, std_dev, true);
    //   3. Update weights tensor
    // Hint: Use Tensor::random_normal() with calculated std
}

void Dense::he_init(int fan_in) {
    // TODO: Implement He initialization
    // Algorithm:
    //   1. Calculate standard deviation: sqrt(2.0 / fan_in)
    double std_dev = std::sqrt(2.0 / fan_in);
    //   2. Generate random weights from normal distribution
    std::vector<int> weight_shape = {output_size_, input_size_};
    weights_ = Tensor::random_normal(weight_shape, 0.0, std_dev, true);
    //   3. Update weights tensor
    // Hint: Use Tensor::random_normal() with calculated std
}

void Dense::uniform_init(double min, double max) {
    // TODO: Implement uniform initialization
    // Algorithm:
    //   1. Generate random weights from uniform distribution
    std::vector<int> weight_shape = {output_size_, input_size_};
    weights_ = Tensor::random(weight_shape, min, max, true);
    //   2. Update weights tensor
    // Hint: Use Tensor::random() with min and max values
}

void Dense::normal_init(double mean, double std) {
    // TODO: Implement normal initialization
    // Algorithm:
    //   1. Generate random weights from normal distribution
    std::vector<int> weight_shape = {output_size_, input_size_};
    weights_ = Tensor::random_normal(weight_shape, mean, std, true);
    //   2. Update weights tensor
    // Hint: Use Tensor::random_normal() with mean and std
}

// Weight and bias access
void Dense::set_weights(const Tensor& weights) {
    // TODO: Implement weight setting
    // Algorithm:
    //   1. Validate weights shape matches expected shape
    std::vector<int> expected_shape = {output_size_, input_size_};
    if (weights.shape() != expected_shape) {
        throw std::invalid_argument("Weight shape mismatch: expected [" + 
                                  std::to_string(output_size_) + ", " + 
                                  std::to_string(input_size_) + "], got different shape");
    }
    //   2. Copy weights to weights_ member
    weights_ = weights;
    //   3. Update weight_gradients_ shape if needed
    weight_gradients_ = Tensor::zeros(expected_shape, false);
    // Hint: Check weights.shape() == [output_size, input_size]
}

void Dense::set_biases(const Tensor& biases) {
    // TODO: Implement bias setting
    // Algorithm:
    //   1. Validate biases shape matches expected shape
    std::vector<int> expected_shape = {output_size_};
    if (biases.shape() != expected_shape) {
        throw std::invalid_argument("Bias shape mismatch: expected [" + 
                                  std::to_string(output_size_) + "], got different shape");
    }
    //   2. Copy biases to biases_ member
    biases_ = biases;
    //   3. Update bias_gradients_ shape if needed
    bias_gradients_ = Tensor::zeros(expected_shape, false);
    // Hint: Check biases.shape() == [output_size]
}

} // namespace cppvision
