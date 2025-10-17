#ifndef CPPVISION_LAYERS_DENSE_H
#define CPPVISION_LAYERS_DENSE_H

#include "../core/layer.h"
#include "../core/tensor.h"
#include <vector>
#include <string>

namespace cppvision {

/**
 * @brief Fully connected (dense) layer implementation
 * 
 * Performs linear transformation: output = input * weights^T + bias
 * This is the fundamental building block for neural networks.
 */
class Dense : public Layer {
private:
    Tensor weights_;              // Weight matrix [output_size, input_size]
    Tensor biases_;                // Bias vector [output_size]
    Tensor weight_gradients_;      // Gradients for weights
    Tensor bias_gradients_;       // Gradients for biases
    Tensor input_cache_;           // Cached input for backward pass
    int input_size_;               // Input dimension
    int output_size_;              // Output dimension
    bool use_bias_;                // Whether to use bias
    std::string name_;             // Layer name

public:
    /**
     * @brief Constructor for dense layer
     * @param input_size Number of input features
     * @param output_size Number of output features
     * @param use_bias Whether to use bias term
     * @param name Optional layer name
     */
    Dense(int input_size, int output_size, bool use_bias = true, const std::string& name = "Dense");

    /**
     * @brief Destructor
     */
    ~Dense() override = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass: Y = X * W^T + b
     * @param input Input tensor [batch_size, input_size]
     * @return Output tensor [batch_size, output_size]
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient from next layer [batch_size, output_size]
     * @return Gradient to previous layer [batch_size, input_size]
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management
    /**
     * @brief Get all trainable parameters
     * @return Vector containing weights and biases (if used)
     */
    std::vector<Tensor> get_parameters() const override;
    
    /**
     * @brief Get gradients for all parameters
     * @return Vector containing weight and bias gradients
     */
    std::vector<Tensor> get_gradients() const override;
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients (weights, biases)
     */
    void update_parameters(const std::vector<Tensor>& gradients) override;
    
    /**
     * @brief Zero out all gradients
     */
    void zero_gradients() override;

    // Layer information
    /**
     * @brief Get layer name
     * @return Layer name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get number of parameters
     * @return Total number of parameters
     */
    int num_parameters() const override;
    
    /**
     * @brief Check if layer has parameters
     * @return True (dense layers always have parameters)
     */
    bool has_parameters() const override { return true; }

    // Weight initialization methods
    /**
     * @brief Initialize weights using Xavier/Glorot initialization
     * @param fan_in Number of input units
     * @param fan_out Number of output units
     */
    void xavier_init(int fan_in, int fan_out);
    
    /**
     * @brief Initialize weights using He initialization
     * @param fan_in Number of input units
     */
    void he_init(int fan_in);
    
    /**
     * @brief Initialize weights using uniform distribution
     * @param min Minimum value
     * @param max Maximum value
     */
    void uniform_init(double min = -0.1, double max = 0.1);
    
    /**
     * @brief Initialize weights using normal distribution
     * @param mean Mean of distribution
     * @param std Standard deviation
     */
    void normal_init(double mean = 0.0, double std = 0.1);

    // Accessors
    /**
     * @brief Get input size
     * @return Input dimension
     */
    int get_input_size() const { return input_size_; }
    
    /**
     * @brief Get output size
     * @return Output dimension
     */
    int get_output_size() const { return output_size_; }
    
    /**
     * @brief Check if bias is used
     * @return True if bias is used
     */
    bool uses_bias() const { return use_bias_; }

    // Weight and bias access
    /**
     * @brief Get weight matrix
     * @return Weight tensor
     */
    const Tensor& get_weights() const { return weights_; }
    
    /**
     * @brief Get bias vector
     * @return Bias tensor
     */
    const Tensor& get_biases() const { return biases_; }
    
    /**
     * @brief Set weights (for loading pre-trained models)
     * @param weights New weight tensor
     */
    void set_weights(const Tensor& weights);
    
    /**
     * @brief Set biases (for loading pre-trained models)
     * @param biases New bias tensor
     */
    void set_biases(const Tensor& biases);
};

} // namespace cppvision

#endif // CPPVISION_LAYERS_DENSE_H
