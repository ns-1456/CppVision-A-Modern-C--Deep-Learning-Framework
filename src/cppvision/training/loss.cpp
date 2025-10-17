#include "cppvision/training/loss.h"
#include "cppvision/core/tensor.h"
#include <algorithm>
#include <cmath>

namespace cppvision {

// CrossEntropyLoss implementation
CrossEntropyLoss::CrossEntropyLoss(double epsilon, const std::string& name)
    : epsilon_(epsilon), name_(name) {
    // Constructor
}

double CrossEntropyLoss::compute_loss(const Tensor& predictions, const Tensor& targets) {
    // Simple cross-entropy loss computation
    double loss = 0.0;
    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            // Simple log loss with clipping
            double pred = std::max(epsilon_, std::min(1.0 - epsilon_, predictions[idx]));
            loss -= targets[idx] * std::log(pred);
        }
    }
    
    return loss / batch_size;
}

Tensor CrossEntropyLoss::compute_gradients(const Tensor& predictions, const Tensor& targets) {
    // Simple gradient computation
    std::vector<int> grad_shape = predictions.shape();
    Tensor grad = Tensor::zeros(grad_shape, predictions.requires_grad());
    
    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            grad[idx] = predictions[idx] - targets[idx];
        }
    }
    
    return grad;
}

// MSELoss implementation
MSELoss::MSELoss(const std::string& name) : name_(name) {
    // Constructor
}

double MSELoss::compute_loss(const Tensor& predictions, const Tensor& targets) {
    // MSE loss computation
    double loss = 0.0;
    int size = predictions.size();
    
    for (int i = 0; i < size; ++i) {
        double diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    
    return loss / size;
}

Tensor MSELoss::compute_gradients(const Tensor& predictions, const Tensor& targets) {
    // MSE gradient computation
    std::vector<int> grad_shape = predictions.shape();
    Tensor grad = Tensor::zeros(grad_shape, predictions.requires_grad());
    
    int size = predictions.size();
    for (int i = 0; i < size; ++i) {
        grad[i] = 2.0 * (predictions[i] - targets[i]);
    }
    
    return grad;
}

} // namespace cppvision