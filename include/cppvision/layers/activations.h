#ifndef CPPVISION_LAYERS_ACTIVATIONS_H
#define CPPVISION_LAYERS_ACTIVATIONS_H

#include "../core/layer.h"
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <cmath>

namespace cppvision {

/**
 * @brief ReLU (Rectified Linear Unit) activation layer
 * 
 * Applies ReLU activation: f(x) = max(0, x)
 * During backward pass, gradients are zeroed for negative inputs.
 */
class ReLU : public Layer {
private:
    Tensor input_cache_;           // Cached input for backward pass
    std::string name_;             // Layer name

public:
    /**
     * @brief Constructor for ReLU layer
     * @param name Optional layer name
     */
    explicit ReLU(const std::string& name = "ReLU");

    /**
     * @brief Destructor
     */
    ~ReLU() override = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass: apply ReLU activation
     * @param input Input tensor
     * @return Output tensor with ReLU applied
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management (ReLU has no parameters)
    /**
     * @brief Get all trainable parameters
     * @return Empty vector (ReLU has no parameters)
     */
    std::vector<Tensor> get_parameters() const override { return {}; }
    
    /**
     * @brief Get gradients for all parameters
     * @return Empty vector (ReLU has no parameters)
     */
    std::vector<Tensor> get_gradients() const override { return {}; }
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients (ignored for ReLU)
     */
    void update_parameters(const std::vector<Tensor>& gradients) override {}
    
    /**
     * @brief Zero out all gradients
     */
    void zero_gradients() override {}

    // Layer information
    /**
     * @brief Get layer name
     * @return Layer name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get number of parameters
     * @return 0 (ReLU has no parameters)
     */
    int num_parameters() const override { return 0; }
    
    /**
     * @brief Check if layer has parameters
     * @return False (ReLU has no parameters)
     */
    bool has_parameters() const override { return false; }
};

/**
 * @brief Softmax activation layer
 * 
 * Applies softmax activation: f(x_i) = exp(x_i) / sum(exp(x_j))
 * Used for multi-class classification to produce probability distributions.
 */
class Softmax : public Layer {
private:
    Tensor output_cache_;          // Cached output for backward pass
    std::string name_;             // Layer name
    int axis_;                     // Axis along which to apply softmax

public:
    /**
     * @brief Constructor for softmax layer
     * @param axis Axis along which to apply softmax (default: -1 for last axis)
     * @param name Optional layer name
     */
    explicit Softmax(int axis = -1, const std::string& name = "Softmax");

    /**
     * @brief Destructor
     */
    ~Softmax() override = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass: apply softmax activation
     * @param input Input tensor
     * @return Output tensor with softmax applied
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management (Softmax has no parameters)
    /**
     * @brief Get all trainable parameters
     * @return Empty vector (softmax has no parameters)
     */
    std::vector<Tensor> get_parameters() const override { return {}; }
    
    /**
     * @brief Get gradients for all parameters
     * @return Empty vector (softmax has no parameters)
     */
    std::vector<Tensor> get_gradients() const override { return {}; }
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients (ignored for softmax)
     */
    void update_parameters(const std::vector<Tensor>& gradients) override {}
    
    /**
     * @brief Zero out all gradients
     */
    void zero_gradients() override {}

    // Layer information
    /**
     * @brief Get layer name
     * @return Layer name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get number of parameters
     * @return 0 (softmax has no parameters)
     */
    int num_parameters() const override { return 0; }
    
    /**
     * @brief Check if layer has parameters
     * @return False (softmax has no parameters)
     */
    bool has_parameters() const override { return false; }

    // Accessors
    /**
     * @brief Get softmax axis
     * @return Axis along which softmax is applied
     */
    int get_axis() const { return axis_; }

private:
    /**
     * @brief Compute softmax along specified axis
     * @param input Input tensor
     * @return Softmax output
     */
    Tensor compute_softmax(const Tensor& input) const;
    
    /**
     * @brief Compute softmax gradient (Jacobian matrix)
     * @param output Softmax output
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor compute_softmax_gradient(const Tensor& output, const Tensor& grad_output) const;
};

/**
 * @brief Sigmoid activation layer
 * 
 * Applies sigmoid activation: f(x) = 1 / (1 + exp(-x))
 * Used for binary classification and gating mechanisms.
 */
class Sigmoid : public Layer {
private:
    Tensor output_cache_;          // Cached output for backward pass
    std::string name_;             // Layer name

public:
    /**
     * @brief Constructor for sigmoid layer
     * @param name Optional layer name
     */
    explicit Sigmoid(const std::string& name = "Sigmoid");

    /**
     * @brief Destructor
     */
    ~Sigmoid() override = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass: apply sigmoid activation
     * @param input Input tensor
     * @return Output tensor with sigmoid applied
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management (Sigmoid has no parameters)
    /**
     * @brief Get all trainable parameters
     * @return Empty vector (sigmoid has no parameters)
     */
    std::vector<Tensor> get_parameters() const override { return {}; }
    
    /**
     * @brief Get gradients for all parameters
     * @return Empty vector (sigmoid has no parameters)
     */
    std::vector<Tensor> get_gradients() const override { return {}; }
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients (ignored for sigmoid)
     */
    void update_parameters(const std::vector<Tensor>& gradients) override {}
    
    /**
     * @brief Zero out all gradients
     */
    void zero_gradients() override {}

    // Layer information
    /**
     * @brief Get layer name
     * @return Layer name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get number of parameters
     * @return 0 (sigmoid has no parameters)
     */
    int num_parameters() const override { return 0; }
    
    /**
     * @brief Check if layer has parameters
     * @return False (sigmoid has no parameters)
     */
    bool has_parameters() const override { return false; }
};

/**
 * @brief Tanh activation layer
 * 
 * Applies hyperbolic tangent activation: f(x) = tanh(x)
 * Alternative to sigmoid with zero-centered output.
 */
class Tanh : public Layer {
private:
    Tensor output_cache_;          // Cached output for backward pass
    std::string name_;             // Layer name

public:
    /**
     * @brief Constructor for tanh layer
     * @param name Optional layer name
     */
    explicit Tanh(const std::string& name = "Tanh");

    /**
     * @brief Destructor
     */
    ~Tanh() override = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass: apply tanh activation
     * @param input Input tensor
     * @return Output tensor with tanh applied
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management (Tanh has no parameters)
    /**
     * @brief Get all trainable parameters
     * @return Empty vector (tanh has no parameters)
     */
    std::vector<Tensor> get_parameters() const override { return {}; }
    
    /**
     * @brief Get gradients for all parameters
     * @return Empty vector (tanh has no parameters)
     */
    std::vector<Tensor> get_gradients() const override { return {}; }
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients (ignored for tanh)
     */
    void update_parameters(const std::vector<Tensor>& gradients) override {}
    
    /**
     * @brief Zero out all gradients
     */
    void zero_gradients() override {}

    // Layer information
    /**
     * @brief Get layer name
     * @return Layer name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get number of parameters
     * @return 0 (tanh has no parameters)
     */
    int num_parameters() const override { return 0; }
    
    /**
     * @brief Check if layer has parameters
     * @return False (tanh has no parameters)
     */
    bool has_parameters() const override { return false; }
};

} // namespace cppvision

#endif // CPPVISION_LAYERS_ACTIVATIONS_H
