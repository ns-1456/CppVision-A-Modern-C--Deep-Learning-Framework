#ifndef CPPVISION_CORE_TENSOR_H
#define CPPVISION_CORE_TENSOR_H

#include <vector>
#include <string>
#include <random>
#include <iostream>

namespace cppvision {

/**
 * @brief Multi-dimensional tensor class for neural network computations
 * 
 * The Tensor class provides a foundation for all neural network operations.
 * It stores data as a flattened vector with shape information for multi-dimensional access.
 * Supports basic mathematical operations, shape manipulation, and memory-efficient storage.
 */
class Tensor {
private:
    std::vector<double> data_;           // Flattened data storage
    std::vector<int> shape_;             // Shape dimensions [height, width, channels, ...]
    bool requires_grad_;                 // Whether gradients should be computed
    std::string name_;                   // Optional name for debugging

public:
    // Constructors
    /**
     * @brief Default constructor - creates empty tensor
     */
    Tensor() = default;
    
    /**
     * @brief Constructor with shape specification
     * @param shape Vector of dimensions [dim0, dim1, dim2, ...]
     * @param requires_grad Whether gradients should be tracked
     * @param name Optional name for debugging
     */
    Tensor(const std::vector<int>& shape, bool requires_grad = false, const std::string& name = "");
    
    /**
     * @brief Constructor with data and shape
     * @param data Flattened data vector
     * @param shape Vector of dimensions
     * @param requires_grad Whether gradients should be tracked
     * @param name Optional name for debugging
     */
    Tensor(const std::vector<double>& data, const std::vector<int>& shape, 
           bool requires_grad = false, const std::string& name = "");

    // Destructor
    ~Tensor() = default;

    // Copy and move constructors/operators
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Accessors and mutators
    /**
     * @brief Get the shape of the tensor
     * @return Vector of dimensions
     */
    const std::vector<int>& shape() const { return shape_; }
    
    /**
     * @brief Get the total number of elements
     * @return Total size
     */
    int size() const;
    
    /**
     * @brief Get the number of dimensions
     * @return Number of dimensions
     */
    int ndim() const { return static_cast<int>(shape_.size()); }
    
    /**
     * @brief Check if gradients are required
     * @return True if gradients should be computed
     */
    bool requires_grad() const { return requires_grad_; }
    
    /**
     * @brief Set gradient requirement
     * @param requires_grad Whether gradients should be tracked
     */
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    /**
     * @brief Get the name of the tensor
     * @return Name string
     */
    const std::string& name() const { return name_; }
    
    /**
     * @brief Set the name of the tensor
     * @param name Name string
     */
    void set_name(const std::string& name) { name_ = name; }

    // Data access
    /**
     * @brief Access element by multi-dimensional indices
     * @param indices Vector of indices for each dimension
     * @return Reference to the element
     */
    double& operator()(const std::vector<int>& indices);
    
    /**
     * @brief Access element by multi-dimensional indices (const version)
     * @param indices Vector of indices for each dimension
     * @return Const reference to the element
     */
    const double& operator()(const std::vector<int>& indices) const;
    
    /**
     * @brief Access element by flattened index
     * @param index Flattened index
     * @return Reference to the element
     */
    double& operator[](int index);
    
    /**
     * @brief Access element by flattened index (const version)
     * @param index Flattened index
     * @return Const reference to the element
     */
    const double& operator[](int index) const;
    
    /**
     * @brief Get raw data pointer
     * @return Pointer to data
     */
    double* data() { return data_.data(); }
    
    /**
     * @brief Get raw data pointer (const version)
     * @return Const pointer to data
     */
    const double* data() const { return data_.data(); }

    // Shape operations
    /**
     * @brief Reshape the tensor to new dimensions
     * @param new_shape New shape dimensions
     * @return New tensor with reshaped data
     */
    Tensor reshape(const std::vector<int>& new_shape) const;
    
    /**
     * @brief Transpose the tensor (swap last two dimensions)
     * @return New transposed tensor
     */
    Tensor transpose() const;
    
    /**
     * @brief Flatten the tensor to 1D
     * @return New flattened tensor
     */
    Tensor flatten() const;
    
    /**
     * @brief Squeeze dimensions of size 1
     * @return New tensor with squeezed dimensions
     */
    Tensor squeeze() const;
    
    /**
     * @brief Unsqueeze by adding dimension of size 1 at specified axis
     * @param axis Axis to add dimension at
     * @return New tensor with added dimension
     */
    Tensor unsqueeze(int axis) const;

    // Mathematical operations
    /**
     * @brief Element-wise addition
     * @param other Tensor to add
     * @return New tensor with result
     */
    Tensor operator+(const Tensor& other) const;
    
    /**
     * @brief Element-wise subtraction
     * @param other Tensor to subtract
     * @return New tensor with result
     */
    Tensor operator-(const Tensor& other) const;
    
    /**
     * @brief Element-wise multiplication
     * @param other Tensor to multiply
     * @return New tensor with result
     */
    Tensor operator*(const Tensor& other) const;
    
    /**
     * @brief Scalar multiplication
     * @param scalar Scalar value
     * @return New tensor with result
     */
    Tensor operator*(double scalar) const;
    
    /**
     * @brief Matrix multiplication (last two dimensions)
     * @param other Tensor to multiply with
     * @return New tensor with result
     */
    Tensor matmul(const Tensor& other) const;
    
    /**
     * @brief Dot product (for 1D tensors)
     * @param other Tensor to dot with
     * @return Scalar result
     */
    double dot(const Tensor& other) const;

    // Utility functions
    /**
     * @brief Create tensor filled with zeros
     * @param shape Shape of the tensor
     * @param requires_grad Whether gradients should be tracked
     * @return New zero tensor
     */
    static Tensor zeros(const std::vector<int>& shape, bool requires_grad = false);
    
    /**
     * @brief Create tensor filled with ones
     * @param shape Shape of the tensor
     * @param requires_grad Whether gradients should be tracked
     * @return New ones tensor
     */
    static Tensor ones(const std::vector<int>& shape, bool requires_grad = false);
    
    /**
     * @brief Create tensor with random values (uniform distribution)
     * @param shape Shape of the tensor
     * @param min Minimum value
     * @param max Maximum value
     * @param requires_grad Whether gradients should be tracked
     * @return New random tensor
     */
    static Tensor random(const std::vector<int>& shape, double min = 0.0, double max = 1.0, 
                        bool requires_grad = false);
    
    /**
     * @brief Create tensor with random values (normal distribution)
     * @param shape Shape of the tensor
     * @param mean Mean of distribution
     * @param std Standard deviation
     * @param requires_grad Whether gradients should be tracked
     * @return New random tensor
     */
    static Tensor random_normal(const std::vector<int>& shape, double mean = 0.0, double std = 1.0,
                               bool requires_grad = false);

    // Debugging and visualization
    /**
     * @brief Print tensor information
     * @param os Output stream
     * @param tensor Tensor to print
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
    /**
     * @brief Print tensor data in formatted way
     */
    void print() const;
    
    /**
     * @brief Get string representation of tensor
     * @return String representation
     */
    std::string to_string() const;

private:
    /**
     * @brief Convert multi-dimensional indices to flattened index
     * @param indices Multi-dimensional indices
     * @return Flattened index
     */
    int indices_to_flat(const std::vector<int>& indices) const;
    
    /**
     * @brief Convert flattened index to multi-dimensional indices
     * @param flat_index Flattened index
     * @return Multi-dimensional indices
     */
    std::vector<int> flat_to_indices(int flat_index) const;
    
    /**
     * @brief Validate that indices are within bounds
     * @param indices Multi-dimensional indices
     * @return True if valid
     */
    bool is_valid_indices(const std::vector<int>& indices) const;
    
    /**
     * @brief Calculate total size from shape
     * @param shape Shape vector
     * @return Total size
     */
    static int calculate_size(const std::vector<int>& shape);
};

} // namespace cppvision

#endif // CPPVISION_CORE_TENSOR_H
