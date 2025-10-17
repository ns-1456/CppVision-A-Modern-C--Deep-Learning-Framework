#ifndef CPPVISION_OPTIMIZERS_SGD_H
#define CPPVISION_OPTIMIZERS_SGD_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <map>

namespace cppvision {

/**
 * @brief Stochastic Gradient Descent optimizer with momentum
 * 
 * Implements SGD with optional momentum and weight decay.
 * Updates parameters using: param = param - lr * (grad + momentum * velocity)
 */
class SGD : public Optimizer {
private:
    double learning_rate_;         // Learning rate
    double momentum_;             // Momentum factor
    double weight_decay_;         // Weight decay factor
    std::string name_;            // Optimizer name
    
    // Momentum state
    std::vector<Tensor> velocities_;  // Velocity vectors for momentum
    bool momentum_initialized_;      // Whether momentum is initialized

public:
    /**
     * @brief Constructor for SGD optimizer
     * @param learning_rate Learning rate
     * @param momentum Momentum factor (0.0 to disable momentum)
     * @param weight_decay Weight decay factor (0.0 to disable weight decay)
     * @param name Optimizer name
     */
    explicit SGD(double learning_rate = 0.01, double momentum = 0.0, 
                double weight_decay = 0.0, const std::string& name = "SGD");

    /**
     * @brief Destructor
     */
    ~SGD() override = default;

    /**
     * @brief Update parameters using gradients
     * @param parameters Vector of parameter tensors
     * @param gradients Vector of gradient tensors
     */
    void step(const std::vector<Tensor>& parameters, 
             const std::vector<Tensor>& gradients) override;
    
    /**
     * @brief Zero out gradients
     * @param gradients Vector of gradient tensors
     */
    void zero_grad(std::vector<Tensor>& gradients) override;
    
    /**
     * @brief Get optimizer name
     * @return Optimizer name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get learning rate
     * @return Current learning rate
     */
    double get_learning_rate() const override { return learning_rate_; }
    
    /**
     * @brief Set learning rate
     * @param lr New learning rate
     */
    void set_learning_rate(double lr) override { learning_rate_ = lr; }
    
    /**
     * @brief Get optimizer state
     * @return Optimizer state map
     */
    std::map<std::string, double> get_state() const override;
    
    /**
     * @brief Set optimizer state
     * @param state Optimizer state map
     */
    void set_state(const std::map<std::string, double>& state) override;
    
    /**
     * @brief Reset optimizer state
     */
    void reset_state() override;

    // SGD-specific methods
    /**
     * @brief Get momentum factor
     * @return Momentum factor
     */
    double get_momentum() const { return momentum_; }
    
    /**
     * @brief Set momentum factor
     * @param momentum New momentum factor
     */
    void set_momentum(double momentum) { momentum_ = momentum; }
    
    /**
     * @brief Get weight decay factor
     * @return Weight decay factor
     */
    double get_weight_decay() const { return weight_decay_; }
    
    /**
     * @brief Set weight decay factor
     * @param weight_decay New weight decay factor
     */
    void set_weight_decay(double weight_decay) { weight_decay_ = weight_decay; }
    
    /**
     * @brief Get velocity vectors
     * @return Vector of velocity tensors
     */
    const std::vector<Tensor>& get_velocities() const { return velocities_; }

private:
    /**
     * @brief Initialize momentum vectors
     * @param parameters Vector of parameter tensors
     */
    void initialize_momentum(const std::vector<Tensor>& parameters);
    
    /**
     * @brief Apply weight decay to gradients
     * @param gradients Vector of gradient tensors
     * @param parameters Vector of parameter tensors
     */
    void apply_weight_decay(std::vector<Tensor>& gradients, 
                           const std::vector<Tensor>& parameters) const;
    
    /**
     * @brief Update parameter with momentum
     * @param param Parameter tensor
     * @param grad Gradient tensor
     * @param velocity Velocity tensor
     */
    void update_with_momentum(Tensor& param, const Tensor& grad, Tensor& velocity) const;
};

/**
 * @brief SGD with Nesterov momentum
 * 
 * Implements SGD with Nesterov accelerated gradient.
 * Uses lookahead gradient computation for better convergence.
 */
class SGDNesterov : public SGD {
private:
    std::string name_;            // Optimizer name

public:
    /**
     * @brief Constructor for SGD with Nesterov momentum
     * @param learning_rate Learning rate
     * @param momentum Momentum factor
     * @param weight_decay Weight decay factor
     * @param name Optimizer name
     */
    explicit SGDNesterov(double learning_rate = 0.01, double momentum = 0.9,
                        double weight_decay = 0.0, 
                        const std::string& name = "SGDNesterov");

    /**
     * @brief Destructor
     */
    ~SGDNesterov() override = default;

    /**
     * @brief Update parameters using gradients with Nesterov momentum
     * @param parameters Vector of parameter tensors
     * @param gradients Vector of gradient tensors
     */
    void step(const std::vector<Tensor>& parameters, 
             const std::vector<Tensor>& gradients) override;
    
    /**
     * @brief Get optimizer name
     * @return Optimizer name
     */
    std::string get_name() const override { return name_; }

private:
    /**
     * @brief Update parameter with Nesterov momentum
     * @param param Parameter tensor
     * @param grad Gradient tensor
     * @param velocity Velocity tensor
     */
    void update_with_nesterov_momentum(Tensor& param, const Tensor& grad, Tensor& velocity) const;
};

} // namespace cppvision

#endif // CPPVISION_OPTIMIZERS_SGD_H
