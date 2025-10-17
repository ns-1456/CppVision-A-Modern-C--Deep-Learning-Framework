#include "cppvision/core/tensor.h"
#include <iostream>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>

namespace cppvision {

// Constructors
Tensor::Tensor(const std::vector<int>& shape, bool requires_grad, const std::string& name)
    : shape_(shape), requires_grad_(requires_grad), name_(name) {
    // Constructor implementation
    // Algorithm:
    //   1. Validate shape (all dimensions must be positive)
    for (int dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument("Shape dimension must be postive");
        }
    }
    //   2. Calculate total size from shape
    int size = calculate_size(shape);
    //   3. Initialize data_ vector with zeros
    std::vector<double> data(size, 0.0f);

    //   4. Set shape_ and other member variables
    shape_ = shape;
    data_ = data;
    requires_grad_ = requires_grad;
    name_ = name;
    // Hint: Use calculate_size() helper function
}

Tensor::Tensor(const std::vector<double>& data, const std::vector<int>& shape, 
    bool requires_grad, const std::string& name)
// The initializer list handles copying data, shape, and setting flags efficiently.
: data_(data), shape_(shape), requires_grad_(requires_grad), name_(name) {
    // TODO: Implement constructor with data
    // Algorithm:
// 1. Validate shape and data size match
// This is the only check that must happen after initialization and cannot be in the initializer list.
if (data.size() != calculate_size(shape)){
throw std::invalid_argument("Error: Data size (" + std::to_string(data.size()) + 
                         ") must match total size calculated from shape (" + 
                         std::to_string(calculate_size(shape)) + ").");
}
}

// Accessors
int Tensor::size() const {
    // TODO: Implement size calculation
    // Algorithm:
    //   1. Calculate total size from shape dimensions
    int size = calculate_size(shape_);
    //   2. Return the total number of elements
    return size;
    // Hint: Use calculate_size() helper function
}

// Data access
double& Tensor::operator()(const std::vector<int>& indices) {
    // TODO: Implement multi-dimensional indexing
    // Algorithm:
    //   1. Validate indices are within bounds
    if (!is_valid_indices(indices)){
        throw std::invalid_argument("Indices are out of bounds");
    }
    //   2. Convert multi-dimensional indices to flattened index
    int flat_index = indices_to_flat(indices);
    //   3. Return reference to data element
    return data_[flat_index];
    // Hint: Use indices_to_flat() helper function
}

const double& Tensor::operator()(const std::vector<int>& indices) const {
    // TODO: Implement const multi-dimensional indexing
    // Algorithm:
    //   1. Validate indices are within bounds
    if (!is_valid_indices(indices)){
        throw std::invalid_argument("Indices are out of bounds");
    }
    //   2. Convert multi-dimensional indices to flattened index
    int flat_index = indices_to_flat(indices);
    //   3. Return const reference to data element
    return data_[flat_index];
    // Hint: Use indices_to_flat() helper function
}

double& Tensor::operator[](int index) {
    // TODO: Implement flattened indexing
    // Algorithm:
    //   1. Validate index is within bounds
    if (index < 0 || index >= size()) {
        throw std::out_of_range("Index " + std::to_string(index) + " is out of bounds [0, " + std::to_string(size()) + ")");
    }
    //   2. Return reference to data element at index
    return data_[index];
}

const double& Tensor::operator[](int index) const {
    // TODO: Implement const flattened indexing
    // Algorithm:
    //   1. Validate index is within bounds
    if (index < 0 || index >= size()) {
        throw std::out_of_range("Index " + std::to_string(index) + " is out of bounds [0, " + std::to_string(size()) + ")");
    }
    //   2. Return const reference to data element at index
    return data_[index];
}

// Shape operations
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    // TODO: Implement tensor reshaping
    // Algorithm:
    //   1. Validate new_shape has same total size as current shape
    if (calculate_size(new_shape) != size()) {
        throw std::invalid_argument("Cannot reshape tensor: new shape has different total size");
    }
    //   2. Create new tensor with same data but new shape
    Tensor result = *this;  // Copy current tensor
    result.shape_ = new_shape;
    //   3. Return new tensor
    return result;
    // Hint: Check that calculate_size(new_shape) == calculate_size(shape_)
}

Tensor Tensor::transpose() const {
    // TODO: Implement tensor transpose
    // Algorithm:
    //   1. Check if tensor has at least 2 dimensions
    if (ndim() < 2) {
        throw std::invalid_argument("Cannot transpose tensor with less than 2 dimensions");
    }
    //   2. Swap last two dimensions
    std::vector<int> new_shape = shape_;
    std::swap(new_shape[ndim()-2], new_shape[ndim()-1]);
    //   3. Create new tensor with transposed data
    Tensor result = Tensor(new_shape, requires_grad_, name_ + "_transposed");
    //   4. Copy data with transposed indexing
    for (int i = 0; i < size(); ++i) {
        std::vector<int> old_indices = flat_to_indices(i);
        std::swap(old_indices[ndim()-2], old_indices[ndim()-1]);
        int new_flat_index = result.indices_to_flat(old_indices);
        result.data_[new_flat_index] = data_[i];
    }
    //   5. Return new tensor
    return result;
    // Hint: For 2D tensor, swap rows and columns
}

Tensor Tensor::flatten() const {
    // TODO: Implement tensor flattening
    // Algorithm:
    //   1. Calculate total size
    int total_size = size();
    //   2. Create new tensor with shape [total_size]
    std::vector<int> flat_shape = {total_size};
    //   3. Copy data to new tensor
    Tensor result = Tensor(data_, flat_shape, requires_grad_, name_ + "_flattened");
    //   4. Return flattened tensor
    return result;
}

Tensor Tensor::squeeze() const {
    // TODO: Implement dimension squeezing
    // Algorithm:
    //   1. Create new shape without dimensions of size 1
    std::vector<int> new_shape;
    for (int dim : shape_) {
        if (dim != 1) {
            new_shape.push_back(dim);
        }
    }
    //   2. Create new tensor with squeezed shape
    Tensor result = Tensor(data_, new_shape, requires_grad_, name_ + "_squeezed");
    //   3. Copy data to new tensor
    //   4. Return squeezed tensor
    return result;
}

Tensor Tensor::unsqueeze(int axis) const {
    // TODO: Implement dimension unsqueezing
    // Algorithm:
    //   1. Validate axis is within valid range
    if (axis < 0 || axis > ndim()) {
        throw std::invalid_argument("Axis " + std::to_string(axis) + " is out of range [0, " + std::to_string(ndim()) + "]");
    }
    //   2. Insert dimension of size 1 at specified axis
    std::vector<int> new_shape = shape_;
    new_shape.insert(new_shape.begin() + axis, 1);
    //   3. Create new tensor with unsqueezed shape
    Tensor result = Tensor(data_, new_shape, requires_grad_, name_ + "_unsqueezed");
    //   4. Copy data to new tensor
    //   5. Return unsqueezed tensor
    return result;
}

// Mathematical operations
Tensor Tensor::operator+(const Tensor& other) const {
    // TODO: Implement element-wise addition
    // Algorithm:
    //   1. Check if shapes are compatible (broadcasting)
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for element-wise addition");
    }
    //   2. Create new tensor with result shape
    Tensor result = Tensor(shape_, requires_grad_ || other.requires_grad_, name_ + "_add");
    //   3. Perform element-wise addition
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    //   4. Return result tensor
    return result;
    // Hint: Handle broadcasting for different shapes
}

Tensor Tensor::operator-(const Tensor& other) const {
    // TODO: Implement element-wise subtraction
    // Algorithm:
    //   1. Check if shapes are compatible (broadcasting)
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for element-wise subtraction");
    }
    //   2. Create new tensor with result shape
    Tensor result = Tensor(shape_, requires_grad_ || other.requires_grad_, name_ + "_sub");
    //   3. Perform element-wise subtraction
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    //   4. Return result tensor
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    // TODO: Implement element-wise multiplication
    // Algorithm:
    //   1. Check if shapes are compatible (broadcasting)
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    //   2. Create new tensor with result shape
    Tensor result = Tensor(shape_, requires_grad_ || other.requires_grad_, name_ + "_mul");
    //   3. Perform element-wise multiplication
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    //   4. Return result tensor
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    // TODO: Implement scalar multiplication
    // Algorithm:
    //   1. Create new tensor with same shape
    Tensor result = Tensor(shape_, requires_grad_, name_ + "_scalar_mul");
    //   2. Multiply each element by scalar
    for (int i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    //   3. Return result tensor
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // TODO: Implement matrix multiplication
    // Algorithm:
    //   1. Check if last two dimensions are compatible for matrix multiplication
    if (ndim() < 2 || other.ndim() < 2) {
        throw std::invalid_argument("Both tensors must have at least 2 dimensions for matrix multiplication");
    }
    if (shape_[ndim()-1] != other.shape_[other.ndim()-2]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }
    //   2. Create new tensor with result shape
    std::vector<int> result_shape;
    // Handle batch dimensions (all but last 2)
    int batch_dims = std::max(ndim() - 2, other.ndim() - 2);
    for (int i = 0; i < batch_dims; ++i) {
        int dim1 = (i < ndim() - 2) ? shape_[i] : 1;
        int dim2 = (i < other.ndim() - 2) ? other.shape_[i] : 1;
        result_shape.push_back(std::max(dim1, dim2));
    }
    result_shape.push_back(shape_[ndim()-2]);  // Rows of first matrix
    result_shape.push_back(other.shape_[other.ndim()-1]);  // Cols of second matrix
    
    Tensor result = Tensor(result_shape, requires_grad_ || other.requires_grad_, name_ + "_matmul");
    //   3. Perform matrix multiplication for each batch
    // Simplified implementation for 2D case
    if (ndim() == 2 && other.ndim() == 2) {
        int m = shape_[0], n = shape_[1], p = other.shape_[1];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += data_[i * n + k] * other.data_[k * p + j];
                }
                result.data_[i * p + j] = sum;
            }
        }
    }
    //   4. Return result tensor
    return result;
    // Hint: For batch matrix multiplication, iterate over batch dimension
}

double Tensor::dot(const Tensor& other) const {
    // TODO: Implement dot product
    // Algorithm:
    //   1. Check if tensors are 1D and same size
    if (ndim() != 1 || other.ndim() != 1) {
        throw std::invalid_argument("Dot product requires 1D tensors");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Tensors must have same size for dot product");
    }
    //   2. Compute dot product: sum(a[i] * b[i])
    double result = 0.0;
    for (int i = 0; i < size(); ++i) {
        result += data_[i] * other.data_[i];
    }
    //   3. Return scalar result
    return result;
}

// Utility functions
Tensor Tensor::zeros(const std::vector<int>& shape, bool requires_grad) {
    // TODO: Implement zeros tensor creation
    // Algorithm:
    //   1. Calculate total size from shape
    int total_size = calculate_size(shape);
    //   2. Create data vector filled with zeros
    std::vector<double> data(total_size, 0.0);
    //   3. Create and return new tensor
    return Tensor(data, shape, requires_grad, "zeros");
}

Tensor Tensor::ones(const std::vector<int>& shape, bool requires_grad) {
    // TODO: Implement ones tensor creation
    // Algorithm:
    //   1. Calculate total size from shape
    int total_size = calculate_size(shape);
    //   2. Create data vector filled with ones
    std::vector<double> data(total_size, 1.0);
    //   3. Create and return new tensor
    return Tensor(data, shape, requires_grad, "ones");
}

Tensor Tensor::random(const std::vector<int>& shape, double min, double max, bool requires_grad) {
    // TODO: Implement random tensor creation (uniform distribution)
    // Algorithm:
    //   1. Calculate total size from shape
    int total_size = calculate_size(shape);
    //   2. Generate random numbers in [min, max] range
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    
    std::vector<double> data(total_size);
    for (int i = 0; i < total_size; ++i) {
        data[i] = dis(gen);
    }
    //   3. Create and return new tensor
    return Tensor(data, shape, requires_grad, "random");
    // Hint: Use std::uniform_real_distribution
}

Tensor Tensor::random_normal(const std::vector<int>& shape, double mean, double std, bool requires_grad) {
    // TODO: Implement random tensor creation (normal distribution)
    // Algorithm:
    //   1. Calculate total size from shape
    int total_size = calculate_size(shape);
    //   2. Generate random numbers from normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, std);
    
    std::vector<double> data(total_size);
    for (int i = 0; i < total_size; ++i) {
        data[i] = dis(gen);
    }
    //   3. Create and return new tensor
    return Tensor(data, shape, requires_grad, "random_normal");
    // Hint: Use std::normal_distribution
}

// Debugging and visualization
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    // TODO: Implement tensor printing
    // Algorithm:
    //   1. Print tensor name and shape
    os << "Tensor(" << tensor.name_ << ", shape=[";
    for (size_t i = 0; i < tensor.shape_.size(); ++i) {
        if (i > 0) os << ", ";
        os << tensor.shape_[i];
    }
    os << "], size=" << tensor.size() << ")";
    //   2. Print tensor data in formatted way
    if (tensor.size() <= 20) {
        os << " data=[";
        for (int i = 0; i < tensor.size(); ++i) {
            if (i > 0) os << ", ";
            os << tensor.data_[i];
        }
        os << "]";
    } else {
        os << " data=[... " << tensor.size() << " elements ...]";
    }
    //   3. Handle large tensors by showing summary
    return os;
}

void Tensor::print() const {
    // TODO: Implement tensor printing
    // Algorithm:
    //   1. Print tensor information (name, shape, size)
    std::cout << *this << std::endl;
    //   2. Print data in readable format
    //   3. Handle multi-dimensional data appropriately
}

std::string Tensor::to_string() const {
    // TODO: Implement string representation
    // Algorithm:
    //   1. Create string with tensor information
    std::ostringstream oss;
    oss << "Tensor(" << name_ << ", shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], size=" << size() << ")";
    //   2. Include shape, size, and sample data
    //   3. Return formatted string
    return oss.str();
}

// Private helper functions
int Tensor::indices_to_flat(const std::vector<int>& indices) const {
    // TODO: Implement index conversion
    // Algorithm:
    //   1. Validate indices are within bounds
    if (!is_valid_indices(indices)) {
        throw std::invalid_argument("Invalid indices for tensor");
    }
    //   2. Convert multi-dimensional indices to flattened index
    //   3. Use row-major order: index = i0*stride0 + i1*stride1 + ...
    int flat_index = 0;
    int stride = 1;
    for (int i = ndim() - 1; i >= 0; --i) {
        flat_index += indices[i] * stride;
        stride *= shape_[i];
    }
    //   4. Return flattened index
    return flat_index;
}

std::vector<int> Tensor::flat_to_indices(int flat_index) const {
    // TODO: Implement index conversion
    // Algorithm:
    //   1. Validate flat_index is within bounds
    if (flat_index < 0 || flat_index >= size()) {
        throw std::out_of_range("Flat index out of bounds");
    }
    //   2. Convert flattened index to multi-dimensional indices
    //   3. Use row-major order conversion
    std::vector<int> indices(ndim());
    int remaining = flat_index;
    for (int i = ndim() - 1; i >= 0; --i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
    }
    //   4. Return multi-dimensional indices
    return indices;
}

bool Tensor::is_valid_indices(const std::vector<int>& indices) const {
    // TODO: Implement index validation
    // Algorithm:
    //   1. Check if indices size matches tensor dimensions
    if (indices.size() != static_cast<size_t>(ndim())) {
        return false;
    }
    //   2. Check if each index is within bounds [0, shape[i])
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            return false;
        }
    }
    //   3. Return true if valid, false otherwise
    return true;
}

int Tensor::calculate_size(const std::vector<int>& shape) {
    // TODO: Implement size calculation
    // Algorithm:
    //   1. Multiply all dimensions together
    int total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    //   2. Return total size
    return total_size;
    // Hint: Use std::accumulate with std::multiplies
}

} // namespace cppvision
