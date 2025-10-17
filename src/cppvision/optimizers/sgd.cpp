#include "cppvision/optimizers/sgd.h"
#include "cppvision/core/tensor.h"
#include <algorithm>

namespace cppvision {

// SGD Constructor
SGD::SGD(double learning_rate, double momentum, double weight_decay, const std::string& name)
    : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay), 
      name_(name), momentum_initialized_(false) {
    // Constructor - initialize optimizer parameters
}

// Parameter update
void SGD::step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    // TODO: Implement SGD parameter update
    // Algorithm:
    //   1. Initialize momentum if not done yet
    if (!momentum_initialized_) {
        initialize_momentum(parameters);
    }
    
    //   2. Apply weight decay to gradients if weight_decay_ > 0
    std::vector<Tensor> modified_gradients = gradients;
    if (weight_decay_ > 0) {
        apply_weight_decay(modified_gradients, parameters);
    }
    
    //   3. Update parameters with momentum
    for (size_t i = 0; i < parameters.size(); ++i) {
        update_with_momentum(const_cast<Tensor&>(parameters[i]), modified_gradients[i], velocities_[i]);
    }
    // Hint: Use update_with_momentum() helper function
}

void SGD::zero_grad(std::vector<Tensor>& gradients) {
    // TODO: Implement gradient zeroing
    // Algorithm:
    //   1. Set all gradient elements to zero
    for (auto& grad : gradients) {
        for (int i = 0; i < grad.size(); ++i) {
            grad[i] = 0.0;
        }
    }
    //   2. Maintain gradient tensor shapes
    // Hint: Use tensor operations to fill with zeros
}

// Optimizer state management
std::map<std::string, double> SGD::get_state() const {
    // TODO: Implement state retrieval
    // Algorithm:
    //   1. Create state map
    //   2. Add learning_rate, momentum, weight_decay
    //   3. Add momentum initialization flag
    //   4. Return state map
    return {}; // Placeholder
}

void SGD::set_state(const std::map<std::string, double>& state) {
    // TODO: Implement state setting
    // Algorithm:
    //   1. Extract learning_rate, momentum, weight_decay from state
    //   2. Update member variables
    //   3. Handle momentum initialization flag
    // Hint: Validate state keys exist
}

void SGD::reset_state() {
    // TODO: Implement state reset
    // Algorithm:
    //   1. Clear velocity vectors
    //   2. Reset momentum initialization flag
    //   3. Reset step counter if exists
    // Hint: Clear velocities_ container
}

// Private helper methods
void SGD::initialize_momentum(const std::vector<Tensor>& parameters) {
    // TODO: Implement momentum initialization
    // Algorithm:
    //   1. Create velocity tensors with same shapes as parameters
    velocities_.clear();
    for (const auto& param : parameters) {
        velocities_.push_back(Tensor::zeros(param.shape(), false));
    }
    //   2. Initialize velocities to zero
    //   3. Set momentum_initialized_ to true
    momentum_initialized_ = true;
    // Hint: Use Tensor::zeros() for velocity initialization
}

void SGD::apply_weight_decay(std::vector<Tensor>& gradients, const std::vector<Tensor>& parameters) const {
    // TODO: Implement weight decay application
    // Algorithm:
    //   1. If weight_decay_ > 0:
    if (weight_decay_ > 0) {
        for (size_t i = 0; i < gradients.size(); ++i) {
            for (int j = 0; j < gradients[i].size(); ++j) {
                gradients[i][j] += weight_decay_ * parameters[i][j];
            }
        }
    }
    //      - Add weight_decay * parameter to each gradient
    //      - This implements L2 regularization
    //   2. Modify gradients in place
    // Hint: gradients[i] += weight_decay_ * parameters[i]
}

void SGD::update_with_momentum(Tensor& param, const Tensor& grad, Tensor& velocity) const {
    // TODO: Implement momentum update
    // Algorithm:
    //   1. Update velocity: velocity = momentum * velocity + learning_rate * grad
    for (int i = 0; i < velocity.size(); ++i) {
        velocity[i] = momentum_ * velocity[i] + learning_rate_ * grad[i];
    }
    //   2. Update parameter: param = param - velocity
    for (int i = 0; i < param.size(); ++i) {
        param[i] -= velocity[i];
    }
    //   3. Handle case where momentum = 0 (pure SGD)
    // Hint: Use tensor operations for element-wise updates
}

// SGD with Nesterov Momentum
SGDNesterov::SGDNesterov(double learning_rate, double momentum, double weight_decay, const std::string& name)
    : SGD(learning_rate, momentum, weight_decay, name), name_(name) {
    // Constructor - inherits from SGD
}

void SGDNesterov::step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    // TODO: Implement Nesterov momentum update
    // Algorithm:
    //   1. Initialize momentum if not done yet
    //   2. Apply weight decay to gradients
    //   3. Update parameters with Nesterov momentum:
    //      - Look ahead: param_ahead = param - momentum * velocity
    //      - Compute gradient at look-ahead position
    //      - velocity = momentum * velocity + learning_rate * grad
    //      - param = param - velocity
    // Hint: Use update_with_nesterov_momentum() helper function
}

void SGDNesterov::update_with_nesterov_momentum(Tensor& param, const Tensor& grad, Tensor& velocity) const {
    // TODO: Implement Nesterov momentum update
    // Algorithm:
    //   1. Update velocity: velocity = momentum * velocity + learning_rate * grad
    //   2. Update parameter: param = param - velocity
    // Note: The look-ahead gradient computation is handled in step()
    // Hint: Same as regular momentum but with look-ahead gradient
}

} // namespace cppvision
