#ifndef CPPVISION_CORE_LAYER_H
#define CPPVISION_CORE_LAYER_H

#include "tensor.h"
#include <vector>
#include <string>
#include <memory>

namespace cppvision {

/**
 * @brief Abstract base class for all neural network layers
 * 
 * The Layer class defines the interface that all neural network layers must implement.
 * It provides the foundation for forward and backward propagation, parameter management,
 * and gradient computation in the CppVision framework.
 */
class Layer {
public:
    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~Layer() = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass through the layer
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * @brief Backward pass through the layer
     * @param grad_output Gradient from the next layer
     * @return Gradient to pass to the previous layer
     */
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // Parameter management
    /**
     * @brief Get all trainable parameters
     * @return Vector of parameter tensors
     */
    virtual std::vector<Tensor> get_parameters() const = 0;
    
    /**
     * @brief Get gradients for all parameters
     * @return Vector of gradient tensors
     */
    virtual std::vector<Tensor> get_gradients() const = 0;
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradient tensors
     */
    virtual void update_parameters(const std::vector<Tensor>& gradients) = 0;
    
    /**
     * @brief Zero out all gradients
     */
    virtual void zero_gradients() = 0;

    // Layer information
    /**
     * @brief Get the name of the layer
     * @return Layer name
     */
    virtual std::string get_name() const = 0;
    
    /**
     * @brief Get the number of parameters
     * @return Total number of parameters
     */
    virtual int num_parameters() const = 0;
    
    /**
     * @brief Check if layer has parameters
     * @return True if layer has trainable parameters
     */
    virtual bool has_parameters() const = 0;

    // Training mode
    /**
     * @brief Set training mode
     * @param training True for training mode, false for evaluation mode
     */
    virtual void set_training(bool training) { training_mode_ = training; }
    
    /**
     * @brief Check if in training mode
     * @return True if in training mode
     */
    bool is_training() const { return training_mode_; }

    // Layer state
    /**
     * @brief Reset layer state (useful for RNNs, BatchNorm, etc.)
     */
    virtual void reset_state() {}

protected:
    bool training_mode_ = true;  // Training mode flag
};

/**
 * @brief Type alias for layer pointer
 */
using LayerPtr = std::unique_ptr<Layer>;

} // namespace cppvision

#endif // CPPVISION_CORE_LAYER_H
