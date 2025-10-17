#include "cppvision/optimizers/adam.h"
#include "cppvision/core/tensor.h"
#include <cmath>
#include <algorithm>

namespace cppvision {

// Adam Constructor
Adam::Adam(double learning_rate, double beta1, double beta2, double epsilon, 
           double weight_decay, const std::string& name)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      weight_decay_(weight_decay), name_(name), step_count_(0), state_initialized_(false) {
    // Constructor - initialize Adam parameters
}

// Parameter update
void Adam::step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    // TODO: Implement Adam parameter update
    // Algorithm:
    //   1. Initialize state if not done yet
    if (!state_initialized_) {
        initialize_state(parameters);
    }
    
    //   2. Increment step counter
    step_count_++;
    
    //   3. Apply weight decay to gradients if weight_decay_ > 0
    std::vector<Tensor> modified_gradients = gradients;
    if (weight_decay_ > 0) {
        apply_weight_decay(modified_gradients, parameters);
    }
    
    //   4. Update first moments: m = beta1 * m + (1 - beta1) * grad
    //   5. Update second moments: v = beta2 * v + (1 - beta2) * grad^2
    //   6. Apply bias correction: m_hat = m / (1 - beta1^step), v_hat = v / (1 - beta2^step)
    //   7. Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    for (size_t i = 0; i < parameters.size(); ++i) {
        update_with_adam(const_cast<Tensor&>(parameters[i]), modified_gradients[i], 
                        first_moments_[i], second_moments_[i]);
    }
    // Hint: Use update_with_adam() helper function
}

void Adam::zero_grad(std::vector<Tensor>& gradients) {
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
std::map<std::string, double> Adam::get_state() const {
    // TODO: Implement state retrieval
    // Algorithm:
    //   1. Create state map
    //   2. Add learning_rate, beta1, beta2, epsilon, weight_decay
    //   3. Add step_count and initialization flag
    //   4. Return state map
    return {}; // Placeholder
}

void Adam::set_state(const std::map<std::string, double>& state) {
    // TODO: Implement state setting
    // Algorithm:
    //   1. Extract parameters from state map
    //   2. Update member variables
    //   3. Handle step_count and initialization flag
    // Hint: Validate state keys exist
}

void Adam::reset_state() {
    // TODO: Implement state reset
    // Algorithm:
    //   1. Clear first and second moment vectors
    //   2. Reset step counter to 0
    //   3. Reset initialization flag
    // Hint: Clear first_moments_ and second_moments_ containers
}

// Private helper methods
void Adam::initialize_state(const std::vector<Tensor>& parameters) {
    // TODO: Implement state initialization
    // Algorithm:
    //   1. Create first moment tensors with same shapes as parameters
    first_moments_.clear();
    for (const auto& param : parameters) {
        first_moments_.push_back(Tensor::zeros(param.shape(), false));
    }
    //   2. Create second moment tensors with same shapes as parameters
    second_moments_.clear();
    for (const auto& param : parameters) {
        second_moments_.push_back(Tensor::zeros(param.shape(), false));
    }
    //   3. Initialize all moments to zero
    //   4. Set state_initialized_ to true
    state_initialized_ = true;
    // Hint: Use Tensor::zeros() for moment initialization
}

void Adam::apply_weight_decay(std::vector<Tensor>& gradients, const std::vector<Tensor>& parameters) const {
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

void Adam::update_with_adam(Tensor& param, const Tensor& grad, 
                           Tensor& first_moment, Tensor& second_moment) const {
    // TODO: Implement Adam update
    // Algorithm:
    //   1. Update first moment: m = beta1 * m + (1 - beta1) * grad
    for (int i = 0; i < first_moment.size(); ++i) {
        first_moment[i] = beta1_ * first_moment[i] + (1.0 - beta1_) * grad[i];
    }
    //   2. Update second moment: v = beta2 * v + (1 - beta2) * grad^2
    for (int i = 0; i < second_moment.size(); ++i) {
        second_moment[i] = beta2_ * second_moment[i] + (1.0 - beta2_) * grad[i] * grad[i];
    }
    //   3. Apply bias correction: m_hat = m / (1 - beta1^step), v_hat = v / (1 - beta2^step)
    double bias_correction1 = bias_correction(beta1_, step_count_);
    double bias_correction2 = bias_correction(beta2_, step_count_);
    
    //   4. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    for (int i = 0; i < param.size(); ++i) {
        double m_hat = first_moment[i] / bias_correction1;
        double v_hat = second_moment[i] / bias_correction2;
        param[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
    // Hint: Use bias_correction() helper function
}

double Adam::bias_correction(double beta, int step) const {
    // TODO: Implement bias correction
    // Algorithm:
    //   1. Compute bias correction factor: 1 / (1 - beta^step)
    double correction = 1.0 / (1.0 - std::pow(beta, step));
    //   2. Return correction factor
    return correction;
    // Hint: Use std::pow() for exponentiation
}

// AdamW Implementation
AdamW::AdamW(double learning_rate, double beta1, double beta2, double epsilon,
             double weight_decay, const std::string& name)
    : Adam(learning_rate, beta1, beta2, epsilon, weight_decay, name), name_(name) {
    // Constructor - inherits from Adam
}

void AdamW::step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    // TODO: Implement AdamW parameter update
    // Algorithm:
    //   1. Call parent Adam step (without weight decay)
    //   2. Apply decoupled weight decay to parameters
    //   3. Weight decay is applied directly to parameters, not gradients
    // Hint: Call Adam::step() then apply_decoupled_weight_decay()
}

void AdamW::apply_decoupled_weight_decay(std::vector<Tensor>& parameters) const {
    // TODO: Implement decoupled weight decay
    // Algorithm:
    //   1. Apply weight decay directly to parameters
    //   2. param = param * (1 - lr * weight_decay)
    //   3. This is different from L2 regularization
    // Hint: Use element-wise operations
}

// RMSprop Implementation
RMSprop::RMSprop(double learning_rate, double alpha, double epsilon, 
                 double weight_decay, double momentum, const std::string& name)
    : learning_rate_(learning_rate), alpha_(alpha), epsilon_(epsilon),
      weight_decay_(weight_decay), momentum_(momentum), name_(name), state_initialized_(false) {
    // Constructor - initialize RMSprop parameters
}

void RMSprop::step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    // TODO: Implement RMSprop parameter update
    // Algorithm:
    //   1. Initialize state if not done yet
    //   2. Apply weight decay to gradients if weight_decay_ > 0
    //   3. Update squared gradients: v = alpha * v + (1 - alpha) * grad^2
    //   4. Update parameters: param = param - lr * grad / (sqrt(v) + epsilon)
    //   5. Apply momentum if momentum_ > 0
    // Hint: Use update_with_rmsprop() helper function
}

void RMSprop::zero_grad(std::vector<Tensor>& gradients) {
    // TODO: Implement gradient zeroing
    // Algorithm:
    //   1. Set all gradient elements to zero
    //   2. Maintain gradient tensor shapes
    // Hint: Use tensor operations to fill with zeros
}

std::map<std::string, double> RMSprop::get_state() const {
    // TODO: Implement state retrieval
    // Algorithm:
    //   1. Create state map with RMSprop parameters
    //   2. Add learning_rate, alpha, epsilon, weight_decay, momentum
    //   3. Add initialization flag
    //   4. Return state map
    return {}; // Placeholder
}

void RMSprop::set_state(const std::map<std::string, double>& state) {
    // TODO: Implement state setting
    // Algorithm:
    //   1. Extract RMSprop parameters from state map
    //   2. Update member variables
    //   3. Handle initialization flag
    // Hint: Validate state keys exist
}

void RMSprop::reset_state() {
    // TODO: Implement state reset
    // Algorithm:
    //   1. Clear squared gradients and velocity vectors
    //   2. Reset initialization flag
    // Hint: Clear squared_gradients_ and velocities_ containers
}

void RMSprop::initialize_state(const std::vector<Tensor>& parameters) {
    // TODO: Implement state initialization
    // Algorithm:
    //   1. Create squared gradient tensors with same shapes as parameters
    //   2. Create velocity tensors if momentum > 0
    //   3. Initialize all to zero
    //   4. Set state_initialized_ to true
    // Hint: Use Tensor::zeros() for initialization
}

void RMSprop::apply_weight_decay(std::vector<Tensor>& gradients, const std::vector<Tensor>& parameters) const {
    // TODO: Implement weight decay application
    // Algorithm:
    //   1. If weight_decay_ > 0:
    //      - Add weight_decay * parameter to each gradient
    //      - This implements L2 regularization
    //   2. Modify gradients in place
    // Hint: gradients[i] += weight_decay_ * parameters[i]
}

void RMSprop::update_with_rmsprop(Tensor& param, const Tensor& grad, 
                                  Tensor& squared_grad, Tensor& velocity) const {
    // TODO: Implement RMSprop update
    // Algorithm:
    //   1. Update squared gradients: v = alpha * v + (1 - alpha) * grad^2
    //   2. Compute update: update = lr * grad / (sqrt(v) + epsilon)
    //   3. Apply momentum if momentum_ > 0:
    //      - velocity = momentum * velocity + update
    //      - param = param - velocity
    //   4. Otherwise: param = param - update
    // Hint: Use tensor operations for element-wise updates
}

} // namespace cppvision
