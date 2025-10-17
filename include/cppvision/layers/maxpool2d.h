#ifndef CPPVISION_LAYERS_MAXPOOL2D_H
#define CPPVISION_LAYERS_MAXPOOL2D_H

#include "../core/layer.h"
#include "../core/tensor.h"
#include <vector>
#include <string>

namespace cppvision {

/**
 * @brief 2D Max Pooling layer implementation
 * 
 * Performs 2D max pooling operation to reduce spatial dimensions.
 * During forward pass, selects maximum value in each pooling window.
 * During backward pass, routes gradients only to the maximum elements.
 */
class MaxPool2D : public Layer {
private:
    Tensor mask_cache_;            // Cached mask for backward pass [batch_size, channels, out_h, out_w]
    Tensor input_cache_;           // Cached input for backward pass
    
    int pool_height_;              // Pooling window height
    int pool_width_;               // Pooling window width
    int stride_h_;                 // Vertical stride
    int stride_w_;                 // Horizontal stride
    int padding_h_;                // Vertical padding
    int padding_w_;                // Horizontal padding
    std::string name_;             // Layer name

public:
    /**
     * @brief Constructor for 2D max pooling layer
     * @param pool_size Pooling window size (assumes square window)
     * @param stride Stride (assumes same for both dimensions)
     * @param padding Padding (assumes same for both dimensions)
     * @param name Optional layer name
     */
    MaxPool2D(int pool_size, int stride = -1, int padding = 0, 
               const std::string& name = "MaxPool2D");
    
    /**
     * @brief Constructor with separate height/width parameters
     * @param pool_h Pooling window height
     * @param pool_w Pooling window width
     * @param stride_h Vertical stride
     * @param stride_w Horizontal stride
     * @param padding_h Vertical padding
     * @param padding_w Horizontal padding
     * @param name Optional layer name
     */
    MaxPool2D(int pool_h, int pool_w, int stride_h = -1, int stride_w = -1,
              int padding_h = 0, int padding_w = 0, 
              const std::string& name = "MaxPool2D");

    /**
     * @brief Destructor
     */
    ~MaxPool2D() override = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass: perform 2D max pooling
     * @param input Input tensor [batch_size, channels, height, width]
     * @return Output tensor [batch_size, channels, out_height, out_width]
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: route gradients through max elements
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management (MaxPool has no parameters)
    /**
     * @brief Get all trainable parameters
     * @return Empty vector (max pooling has no parameters)
     */
    std::vector<Tensor> get_parameters() const override { return {}; }
    
    /**
     * @brief Get gradients for all parameters
     * @return Empty vector (max pooling has no parameters)
     */
    std::vector<Tensor> get_gradients() const override { return {}; }
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients (ignored for max pooling)
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
     * @return 0 (max pooling has no parameters)
     */
    int num_parameters() const override { return 0; }
    
    /**
     * @brief Check if layer has parameters
     * @return False (max pooling has no parameters)
     */
    bool has_parameters() const override { return false; }

    // Accessors
    /**
     * @brief Get pooling window height
     * @return Pool height
     */
    int get_pool_height() const { return pool_height_; }
    
    /**
     * @brief Get pooling window width
     * @return Pool width
     */
    int get_pool_width() const { return pool_width_; }
    
    /**
     * @brief Get vertical stride
     * @return Vertical stride
     */
    int get_stride_h() const { return stride_h_; }
    
    /**
     * @brief Get horizontal stride
     * @return Horizontal stride
     */
    int get_stride_w() const { return stride_w_; }
    
    /**
     * @brief Get vertical padding
     * @return Vertical padding
     */
    int get_padding_h() const { return padding_h_; }
    
    /**
     * @brief Get horizontal padding
     * @return Horizontal padding
     */
    int get_padding_w() const { return padding_w_; }

    // Utility methods
    /**
     * @brief Calculate output height given input height
     * @param input_h Input height
     * @return Output height
     */
    int calculate_output_height(int input_h) const;
    
    /**
     * @brief Calculate output width given input width
     * @param input_w Input width
     * @return Output width
     */
    int calculate_output_width(int input_w) const;

private:
    /**
     * @brief Add padding to input tensor
     * @param input Input tensor
     * @return Padded tensor
     */
    Tensor add_padding(const Tensor& input) const;
    
    /**
     * @brief Remove padding from gradient tensor
     * @param grad_padded Padded gradient tensor
     * @param original_height Original height
     * @param original_width Original width
     * @return Unpadded gradient tensor
     */
    Tensor remove_padding(const Tensor& grad_padded, int original_height, int original_width) const;
    
    /**
     * @brief Find maximum element indices in pooling window
     * @param input Input tensor slice
     * @param start_h Starting height index
     * @param start_w Starting width index
     * @return Pair of (max_value, max_index)
     */
    std::pair<double, int> find_max_in_window(const Tensor& input, int start_h, int start_w) const;
};

} // namespace cppvision

#endif // CPPVISION_LAYERS_MAXPOOL2D_H
