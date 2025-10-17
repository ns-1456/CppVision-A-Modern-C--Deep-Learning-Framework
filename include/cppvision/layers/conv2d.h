#ifndef CPPVISION_LAYERS_CONV2D_H
#define CPPVISION_LAYERS_CONV2D_H

#include "../core/layer.h"
#include "../core/tensor.h"
#include <vector>
#include <string>

namespace cppvision {

/**
 * @brief 2D Convolutional layer implementation
 * 
 * Performs 2D convolution operation on input feature maps.
 * Supports padding, stride, and multiple input/output channels.
 */
class Conv2D : public Layer {
private:
    Tensor kernels_;               // Kernel weights [out_channels, in_channels, kernel_h, kernel_w]
    Tensor biases_;                // Bias vector [out_channels]
    Tensor kernel_gradients_;     // Gradients for kernels
    Tensor bias_gradients_;        // Gradients for biases
    Tensor input_cache_;           // Cached input for backward pass
    Tensor output_cache_;          // Cached output for backward pass
    
    int in_channels_;              // Number of input channels
    int out_channels_;             // Number of output channels
    int kernel_height_;            // Kernel height
    int kernel_width_;             // Kernel width
    int stride_h_;                 // Vertical stride
    int stride_w_;                 // Horizontal stride
    int padding_h_;                // Vertical padding
    int padding_w_;                // Horizontal padding
    bool use_bias_;                // Whether to use bias
    std::string name_;             // Layer name

public:
    /**
     * @brief Constructor for 2D convolutional layer
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Kernel size (assumes square kernel)
     * @param stride Stride (assumes same for both dimensions)
     * @param padding Padding (assumes same for both dimensions)
     * @param use_bias Whether to use bias term
     * @param name Optional layer name
     */
    Conv2D(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0, bool use_bias = true, 
           const std::string& name = "Conv2D");
    
    /**
     * @brief Constructor with separate height/width parameters
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_h Kernel height
     * @param kernel_w Kernel width
     * @param stride_h Vertical stride
     * @param stride_w Horizontal stride
     * @param padding_h Vertical padding
     * @param padding_w Horizontal padding
     * @param use_bias Whether to use bias term
     * @param name Optional layer name
     */
    Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int stride_h = 1, int stride_w = 1, int padding_h = 0, int padding_w = 0,
           bool use_bias = true, const std::string& name = "Conv2D");

    /**
     * @brief Destructor
     */
    ~Conv2D() override = default;

    // Forward and backward propagation
    /**
     * @brief Forward pass: perform 2D convolution
     * @param input Input tensor [batch_size, in_channels, height, width]
     * @return Output tensor [batch_size, out_channels, out_height, out_width]
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * @param grad_output Gradient from next layer
     * @return Gradient to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;

    // Parameter management
    /**
     * @brief Get all trainable parameters
     * @return Vector containing kernels and biases (if used)
     */
    std::vector<Tensor> get_parameters() const override;
    
    /**
     * @brief Get gradients for all parameters
     * @return Vector containing kernel and bias gradients
     */
    std::vector<Tensor> get_gradients() const override;
    
    /**
     * @brief Update parameters using gradients
     * @param gradients Vector of gradients (kernels, biases)
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
     * @return True (conv layers always have parameters)
     */
    bool has_parameters() const override { return true; }

    // Weight initialization methods
    /**
     * @brief Initialize kernels using Xavier/Glorot initialization
     */
    void xavier_init();
    
    /**
     * @brief Initialize kernels using He initialization
     */
    void he_init();
    
    /**
     * @brief Initialize kernels using uniform distribution
     * @param min Minimum value
     * @param max Maximum value
     */
    void uniform_init(double min = -0.1, double max = 0.1);
    
    /**
     * @brief Initialize kernels using normal distribution
     * @param mean Mean of distribution
     * @param std Standard deviation
     */
    void normal_init(double mean = 0.0, double std = 0.1);

    // Accessors
    /**
     * @brief Get number of input channels
     * @return Input channels
     */
    int get_in_channels() const { return in_channels_; }
    
    /**
     * @brief Get number of output channels
     * @return Output channels
     */
    int get_out_channels() const { return out_channels_; }
    
    /**
     * @brief Get kernel height
     * @return Kernel height
     */
    int get_kernel_height() const { return kernel_height_; }
    
    /**
     * @brief Get kernel width
     * @return Kernel width
     */
    int get_kernel_width() const { return kernel_width_; }
    
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
    
    /**
     * @brief Check if bias is used
     * @return True if bias is used
     */
    bool uses_bias() const { return use_bias_; }

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
     * @brief Perform im2col transformation for efficient convolution
     * @param input Input tensor
     * @param batch_size Batch size
     * @param height Input height
     * @param width Input width
     * @return Transformed tensor
     */
    Tensor im2col(const Tensor& input, int batch_size, int height, int width) const;
    
    /**
     * @brief Perform col2im transformation for gradient computation
     * @param grad_col Gradient in column format
     * @param batch_size Batch size
     * @param height Input height
     * @param width Input width
     * @return Transformed gradient tensor
     */
    Tensor col2im(const Tensor& grad_col, int batch_size, int height, int width) const;
    
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
};

} // namespace cppvision

#endif // CPPVISION_LAYERS_CONV2D_H
