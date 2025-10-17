#ifndef CPPVISION_OPTIMIZERS_ADAM_H
#define CPPVISION_OPTIMIZERS_ADAM_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <map>

namespace cppvision {

/**
 * @brief Adam optimizer implementation
 * 
 * Implements the Adam (Adaptive Moment Estimation) optimizer.
 * Combines the benefits of AdaGrad and RMSProp by using both
 * first and second moment estimates with bias correction.
 */
class Adam : public Optimizer {
private:
    double learning_rate_;         // Learning rate
    double beta1_;                // First moment decay rate
    double beta2_;                // Second moment decay rate
    double epsilon_;               // Small value to avoid division by zero
    double weight_decay_;          // Weight decay factor
    std::string name_;             // Optimizer name
    
    // Adam state
    std::vector<Tensor> first_moments_;   // First moment estimates
    std::vector<Tensor> second_moments_;  // Second moment estimates
    int step_count_;                      // Step counter for bias correction
    bool state_initialized_;              // Whether state is initialized

public:
    /**
     * @brief Constructor for Adam optimizer
     * @param learning_rate Learning rate
     * @param beta1 First moment decay rate
     * @param beta2 Second moment decay rate
     * @param epsilon Small value to avoid division by zero
     * @param weight_decay Weight decay factor
     * @param name Optimizer name
     */
    explicit Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999,
                 double epsilon = 1e-8, double weight_decay = 0.0, 
                 const std::string& name = "Adam");

    /**
     * @brief Destructor
     */
    ~Adam() override = default;

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

    // Adam-specific methods
    /**
     * @brief Get beta1 parameter
     * @return Beta1 value
     */
    double get_beta1() const { return beta1_; }
    
    /**
     * @brief Set beta1 parameter
     * @param beta1 New beta1 value
     */
    void set_beta1(double beta1) { beta1_ = beta1; }
    
    /**
     * @brief Get beta2 parameter
     * @return Beta2 value
     */
    double get_beta2() const { return beta2_; }
    
    /**
     * @brief Set beta2 parameter
     * @param beta2 New beta2 value
     */
    void set_beta2(double beta2) { beta2_ = beta2; }
    
    /**
     * @brief Get epsilon parameter
     * @return Epsilon value
     */
    double get_epsilon() const { return epsilon_; }
    
    /**
     * @brief Set epsilon parameter
     * @param epsilon New epsilon value
     */
    void set_epsilon(double epsilon) { epsilon_ = epsilon; }
    
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
     * @brief Get step count
     * @return Current step count
     */
    int get_step_count() const { return step_count_; }
    
    /**
     * @brief Get first moment estimates
     * @return Vector of first moment tensors
     */
    const std::vector<Tensor>& get_first_moments() const { return first_moments_; }
    
    /**
     * @brief Get second moment estimates
     * @return Vector of second moment tensors
     */
    const std::vector<Tensor>& get_second_moments() const { return second_moments_; }

private:
    /**
     * @brief Initialize optimizer state
     * @param parameters Vector of parameter tensors
     */
    void initialize_state(const std::vector<Tensor>& parameters);
    
    /**
     * @brief Apply weight decay to gradients
     * @param gradients Vector of gradient tensors
     * @param parameters Vector of parameter tensors
     */
    void apply_weight_decay(std::vector<Tensor>& gradients, 
                           const std::vector<Tensor>& parameters) const;
    
    /**
     * @brief Update parameter with Adam
     * @param param Parameter tensor
     * @param grad Gradient tensor
     * @param first_moment First moment tensor
     * @param second_moment Second moment tensor
     */
    void update_with_adam(Tensor& param, const Tensor& grad, 
                         Tensor& first_moment, Tensor& second_moment) const;
    
    /**
     * @brief Compute bias correction factor
     * @param beta Decay rate
     * @param step Step number
     * @return Bias correction factor
     */
    double bias_correction(double beta, int step) const;
};

/**
 * @brief AdamW optimizer implementation
 * 
 * Implements AdamW (Adam with decoupled weight decay).
 * Uses decoupled weight decay instead of L2 regularization.
 */
class AdamW : public Adam {
private:
    std::string name_;             // Optimizer name

public:
    /**
     * @brief Constructor for AdamW optimizer
     * @param learning_rate Learning rate
     * @param beta1 First moment decay rate
     * @param beta2 Second moment decay rate
     * @param epsilon Small value to avoid division by zero
     * @param weight_decay Weight decay factor
     * @param name Optimizer name
     */
    explicit AdamW(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999,
                  double epsilon = 1e-8, double weight_decay = 0.01,
                  const std::string& name = "AdamW");

    /**
     * @brief Destructor
     */
    ~AdamW() override = default;

    /**
     * @brief Update parameters using gradients with decoupled weight decay
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
     * @brief Apply decoupled weight decay
     * @param parameters Vector of parameter tensors
     */
    void apply_decoupled_weight_decay(std::vector<Tensor>& parameters) const;
};

/**
 * @brief RMSprop optimizer implementation
 * 
 * Implements RMSprop (Root Mean Square Propagation) optimizer.
 * Uses moving average of squared gradients to normalize the gradient.
 */
class RMSprop : public Optimizer {
private:
    double learning_rate_;         // Learning rate
    double alpha_;                // Smoothing constant
    double epsilon_;              // Small value to avoid division by zero
    double weight_decay_;         // Weight decay factor
    double momentum_;             // Momentum factor
    std::string name_;            // Optimizer name
    
    // RMSprop state
    std::vector<Tensor> squared_gradients_;  // Moving average of squared gradients
    std::vector<Tensor> velocities_;         // Velocity vectors for momentum
    bool state_initialized_;                 // Whether state is initialized

public:
    /**
     * @brief Constructor for RMSprop optimizer
     * @param learning_rate Learning rate
     * @param alpha Smoothing constant
     * @param epsilon Small value to avoid division by zero
     * @param weight_decay Weight decay factor
     * @param momentum Momentum factor
     * @param name Optimizer name
     */
    explicit RMSprop(double learning_rate = 0.01, double alpha = 0.99,
                    double epsilon = 1e-8, double weight_decay = 0.0,
                    double momentum = 0.0, const std::string& name = "RMSprop");

    /**
     * @brief Destructor
     */
    ~RMSprop() override = default;

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

private:
    /**
     * @brief Initialize optimizer state
     * @param parameters Vector of parameter tensors
     */
    void initialize_state(const std::vector<Tensor>& parameters);
    
    /**
     * @brief Apply weight decay to gradients
     * @param gradients Vector of gradient tensors
     * @param parameters Vector of parameter tensors
     */
    void apply_weight_decay(std::vector<Tensor>& gradients, 
                           const std::vector<Tensor>& parameters) const;
    
    /**
     * @brief Update parameter with RMSprop
     * @param param Parameter tensor
     * @param grad Gradient tensor
     * @param squared_grad Squared gradient tensor
     * @param velocity Velocity tensor (if momentum > 0)
     */
    void update_with_rmsprop(Tensor& param, const Tensor& grad, 
                            Tensor& squared_grad, Tensor& velocity) const;
};

} // namespace cppvision

#endif // CPPVISION_OPTIMIZERS_ADAM_H
