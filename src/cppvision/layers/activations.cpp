#include "cppvision/layers/activations.h"
#include "cppvision/core/tensor.h"
#include <cmath>
#include <algorithm>

namespace cppvision {

// ReLU Implementation
ReLU::ReLU(const std::string& name) : name_(name) {
    // Constructor - no parameters needed
}

Tensor ReLU::forward(const Tensor& input) {
    // TODO: Implement ReLU forward pass
    // Algorithm:
    //   1. Cache input for backward pass
    input_cache_ = input;
    //   2. Apply ReLU: output = max(0, input)
    std::vector<double> output_data;
    output_data.reserve(input.size());
    
    for (int i = 0; i < input.size(); ++i) {
        output_data.push_back(std::max(0.0, input[i]));
    }
    //   3. Return output tensor
    Tensor result = Tensor(output_data, input.shape(), input.requires_grad(), name_ + "_relu");
    return result;
    // Hint: Use element-wise maximum operation
}

Tensor ReLU::backward(const Tensor& grad_output) {
    // TODO: Implement ReLU backward pass
    // Algorithm:
    //   1. Create gradient mask: 1.0 where input > 0, 0.0 otherwise
    std::vector<double> grad_input_data;
    grad_input_data.reserve(grad_output.size());
    
    for (int i = 0; i < grad_output.size(); ++i) {
        // Gradient flows only where input > 0, otherwise it's 0
        double mask = (input_cache_[i] > 0.0) ? 1.0 : 0.0;
        grad_input_data.push_back(grad_output[i] * mask);
    }
    //   2. Apply mask to grad_output: grad_input = grad_output * mask
    //   3. Return input gradients
    Tensor result = Tensor(grad_input_data, grad_output.shape(), grad_output.requires_grad(), name_ + "_relu_grad");
    return result;
    // Hint: Use cached input to determine where gradients should flow
}

// Softmax Implementation
Softmax::Softmax(int axis, const std::string& name) : axis_(axis), name_(name) {
    // Constructor - axis determines along which dimension to apply softmax
}

Tensor Softmax::forward(const Tensor& input) {
    // TODO: Implement Softmax forward pass
    // Algorithm:
    //   1. Cache output for backward pass
    Tensor output = compute_softmax(input);
    output_cache_ = output;
    //   2. Apply softmax along specified axis:
    //      - Compute exp(input - max(input)) for numerical stability
    //      - Normalize by sum of exponentials
    //   3. Return softmax probabilities
    return output;
    // Hint: Use compute_softmax() helper function
}

Tensor Softmax::backward(const Tensor& grad_output) {
    // TODO: Implement Softmax backward pass
    // Algorithm:
    //   1. Compute softmax Jacobian matrix
    //   2. Apply Jacobian to grad_output
    Tensor grad_input = compute_softmax_gradient(output_cache_, grad_output);
    //   3. Return input gradients
    return grad_input;
    // Hint: Use compute_softmax_gradient() helper function
}

Tensor Softmax::compute_softmax(const Tensor& input) const {
    // TODO: Implement softmax computation
    // Algorithm:
    //   1. Find maximum value along axis for numerical stability
    double max_val = input[0];
    for (int i = 1; i < input.size(); ++i) {
        max_val = std::max(max_val, input[i]);
    }
    
    //   2. Compute exp(input - max_value)
    std::vector<double> exp_values;
    exp_values.reserve(input.size());
    for (int i = 0; i < input.size(); ++i) {
        exp_values.push_back(std::exp(input[i] - max_val));
    }
    
    //   3. Compute sum of exponentials along axis
    double sum_exp = 0.0;
    for (double val : exp_values) {
        sum_exp += val;
    }
    
    //   4. Normalize: exp_values / sum_exp
    std::vector<double> softmax_values;
    softmax_values.reserve(input.size());
    for (double val : exp_values) {
        softmax_values.push_back(val / sum_exp);
    }
    
    //   5. Return softmax probabilities
    Tensor result = Tensor(softmax_values, input.shape(), input.requires_grad(), name_ + "_softmax");
    return result;
}

Tensor Softmax::compute_softmax_gradient(const Tensor& output, const Tensor& grad_output) const {
    // TODO: Implement softmax gradient computation
    // Algorithm:
    //   1. Compute Jacobian matrix: J[i,j] = output[i] * (delta[i,j] - output[j])
    //   2. Apply Jacobian to grad_output
    // For softmax, the gradient is: grad_input[i] = output[i] * (grad_output[i] - sum(grad_output * output))
    
    // Compute sum(grad_output * output) - this is the dot product
    double dot_product = 0.0;
    for (int i = 0; i < grad_output.size(); ++i) {
        dot_product += grad_output[i] * output[i];
    }
    
    // Compute gradient: grad_input[i] = output[i] * (grad_output[i] - dot_product)
    std::vector<double> grad_input_data;
    grad_input_data.reserve(grad_output.size());
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_input_data.push_back(output[i] * (grad_output[i] - dot_product));
    }
    
    //   3. Return input gradients
    Tensor result = Tensor(grad_input_data, grad_output.shape(), grad_output.requires_grad(), name_ + "_softmax_grad");
    return result;
    // Hint: For softmax, grad_input[i] = output[i] * (grad_output[i] - sum(grad_output * output))
}

// Sigmoid Implementation
Sigmoid::Sigmoid(const std::string& name) : name_(name) {
    // Constructor - no parameters needed
}

Tensor Sigmoid::forward(const Tensor& input) {
    // TODO: Implement Sigmoid forward pass
    // Algorithm:
    //   1. Cache output for backward pass
    std::vector<double> output_data;
    output_data.reserve(input.size());
    
    for (int i = 0; i < input.size(); ++i) {
        output_data.push_back(1.0 / (1.0 + std::exp(-input[i])));
    }
    
    Tensor result = Tensor(output_data, input.shape(), input.requires_grad(), name_ + "_sigmoid");
    output_cache_ = result;
    //   2. Apply sigmoid: output = 1 / (1 + exp(-input))
    //   3. Return output tensor
    return result;
    // Hint: Use std::exp() for exponential function
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    // TODO: Implement Sigmoid backward pass
    // Algorithm:
    //   1. Compute sigmoid gradient: grad_input = grad_output * output * (1 - output)
    std::vector<double> grad_input_data;
    grad_input_data.reserve(grad_output.size());
    
    for (int i = 0; i < grad_output.size(); ++i) {
        double output_val = output_cache_[i];
        grad_input_data.push_back(grad_output[i] * output_val * (1.0 - output_val));
    }
    
    //   2. Return input gradients
    Tensor result = Tensor(grad_input_data, grad_output.shape(), grad_output.requires_grad(), name_ + "_sigmoid_grad");
    return result;
    // Hint: Use cached output to compute gradient efficiently
}

// Tanh Implementation
Tanh::Tanh(const std::string& name) : name_(name) {
    // Constructor - no parameters needed
}

Tensor Tanh::forward(const Tensor& input) {
    // TODO: Implement Tanh forward pass
    // Algorithm:
    //   1. Cache output for backward pass
    std::vector<double> output_data;
    output_data.reserve(input.size());
    
    for (int i = 0; i < input.size(); ++i) {
        output_data.push_back(std::tanh(input[i]));
    }
    
    Tensor result = Tensor(output_data, input.shape(), input.requires_grad(), name_ + "_tanh");
    output_cache_ = result;
    //   2. Apply tanh: output = tanh(input)
    //   3. Return output tensor
    return result;
    // Hint: Use std::tanh() function
}

Tensor Tanh::backward(const Tensor& grad_output) {
    // TODO: Implement Tanh backward pass
    // Algorithm:
    //   1. Compute tanh gradient: grad_input = grad_output * (1 - output^2)
    std::vector<double> grad_input_data;
    grad_input_data.reserve(grad_output.size());
    
    for (int i = 0; i < grad_output.size(); ++i) {
        double output_val = output_cache_[i];
        grad_input_data.push_back(grad_output[i] * (1.0 - output_val * output_val));
    }
    
    //   2. Return input gradients
    Tensor result = Tensor(grad_input_data, grad_output.shape(), grad_output.requires_grad(), name_ + "_tanh_grad");
    return result;
    // Hint: Use cached output to compute gradient efficiently
}

} // namespace cppvision
