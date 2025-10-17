#ifndef CPPVISION_MODELS_SEQUENTIAL_H
#define CPPVISION_MODELS_SEQUENTIAL_H

#include "../core/layer.h"
#include "../core/tensor.h"
#include <vector>
#include <memory>
#include <string>

namespace cppvision {

/**
 * @brief Sequential model container for neural networks
 * 
 * The Sequential model allows stacking layers in a linear sequence.
 * It provides a convenient interface for building feedforward neural networks
 * and handles forward/backward propagation through all layers automatically.
 */
class Sequential : public Layer {
private:
    std::vector<std::unique_ptr<Layer>> layers_;  // Container for layers
    std::string name_;                             // Model name

public:
    /**
     * @brief Default constructor
     * @param name Optional model name
     */
    explicit Sequential(const std::string& name = "Sequential");

    /**
     * @brief Destructor
     */
    ~Sequential() override = default;

    // Layer management
    /**
     * @brief Add a layer to the model
     * @param layer Unique pointer to layer
     */
    void add_layer(std::unique_ptr<Layer> layer);
    
    /**
     * @brief Add a layer using template parameter
     * @tparam LayerType Type of layer to add
     * @tparam Args Constructor argument types
     * @param args Constructor arguments
     */
    template<typename LayerType, typename... Args>
    void add_layer(Args&&... args) {
        add_layer(std::make_unique<LayerType>(std::forward<Args>(args)...));
    }
    
    /**
     * @brief Get number of layers
     * @return Number of layers in the model
     */
    int num_layers() const { return static_cast<int>(layers_.size()); }
    
    /**
     * @brief Get layer by index
     * @param index Layer index
     * @return Reference to layer
     */
    Layer& get_layer(int index);
    
    /**
     * @brief Get layer by index (const version)
     * @param index Layer index
     * @return Const reference to layer
     */
    const Layer& get_layer(int index) const;

    // Forward and backward propagation
    /**
     * @brief Forward pass through all layers
     * @param input Input tensor
     * @return Output tensor from last layer
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass through all layers (in reverse order)
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management
    /**
     * @brief Get all trainable parameters from all layers
     * @return Vector of all parameter tensors
     */
    std::vector<Tensor> get_parameters() const override;
    
    /**
     * @brief Get gradients for all parameters from all layers
     * @return Vector of all gradient tensors
     */
    std::vector<Tensor> get_gradients() const override;
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients for all parameters
     */
    void update_parameters(const std::vector<Tensor>& gradients) override;
    
    /**
     * @brief Zero out all gradients in all layers
     */
    void zero_gradients() override;

    // Layer information
    /**
     * @brief Get model name
     * @return Model name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get total number of parameters across all layers
     * @return Total number of parameters
     */
    int num_parameters() const override;
    
    /**
     * @brief Check if model has parameters
     * @return True if any layer has parameters
     */
    bool has_parameters() const override;

    // Training mode
    /**
     * @brief Set training mode for all layers
     * @param training True for training mode, false for evaluation mode
     */
    void set_training(bool training) override;

    // Model information and utilities
    /**
     * @brief Print model summary
     */
    void summary() const;
    
    /**
     * @brief Get model summary as string
     * @return Model summary string
     */
    std::string get_summary() const;
    
    /**
     * @brief Get input shape requirements
     * @return Vector of input shapes for each layer
     */
    std::vector<std::vector<int>> get_input_shapes() const;
    
    /**
     * @brief Get output shape for given input shape
     * @param input_shape Input tensor shape
     * @return Output tensor shape
     */
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const;

    // Model saving and loading
    /**
     * @brief Save model parameters to file
     * @param filename File path to save to
     */
    void save_parameters(const std::string& filename) const;
    
    /**
     * @brief Load model parameters from file
     * @param filename File path to load from
     */
    void load_parameters(const std::string& filename);

    // Iterator support for range-based loops
    /**
     * @brief Get iterator to beginning of layers
     * @return Iterator to first layer
     */
    auto begin() { return layers_.begin(); }
    
    /**
     * @brief Get iterator to end of layers
     * @return Iterator past last layer
     */
    auto end() { return layers_.end(); }
    
    /**
     * @brief Get const iterator to beginning of layers
     * @return Const iterator to first layer
     */
    auto begin() const { return layers_.begin(); }
    
    /**
     * @brief Get const iterator to end of layers
     * @return Const iterator past last layer
     */
    auto end() const { return layers_.end(); }

private:
    /**
     * @brief Validate model structure
     * @return True if model is valid
     */
    bool validate_model() const;
    
    /**
     * @brief Calculate parameter count for a layer
     * @param layer Layer to count parameters for
     * @return Number of parameters
     */
    int count_layer_parameters(const Layer& layer) const;
};

} // namespace cppvision

#endif // CPPVISION_MODELS_SEQUENTIAL_H
