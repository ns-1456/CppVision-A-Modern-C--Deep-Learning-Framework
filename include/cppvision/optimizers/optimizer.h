#ifndef CPPVISION_OPTIMIZERS_OPTIMIZER_H
#define CPPVISION_OPTIMIZERS_OPTIMIZER_H

#include "../core/tensor.h"
#include <vector>
#include <string>
#include <map>

namespace cppvision {

/**
 * @brief Abstract base class for all optimizers
 * 
 * The Optimizer class defines the interface for all optimization algorithms.
 * It provides methods for parameter updates and gradient management.
 */
class Optimizer {
public:
    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~Optimizer() = default;

    /**
     * @brief Update parameters using gradients
     * @param parameters Vector of parameter tensors
     * @param gradients Vector of gradient tensors
     */
    virtual void step(const std::vector<Tensor>& parameters, 
                     const std::vector<Tensor>& gradients) = 0;
    
    /**
     * @brief Zero out gradients
     * @param gradients Vector of gradient tensors
     */
    virtual void zero_grad(std::vector<Tensor>& gradients) = 0;
    
    /**
     * @brief Get optimizer name
     * @return Optimizer name
     */
    virtual std::string get_name() const = 0;
    
    /**
     * @brief Get learning rate
     * @return Current learning rate
     */
    virtual double get_learning_rate() const = 0;
    
    /**
     * @brief Set learning rate
     * @param lr New learning rate
     */
    virtual void set_learning_rate(double lr) = 0;
    
    /**
     * @brief Get optimizer state
     * @return Optimizer state map
     */
    virtual std::map<std::string, double> get_state() const = 0;
    
    /**
     * @brief Set optimizer state
     * @param state Optimizer state map
     */
    virtual void set_state(const std::map<std::string, double>& state) = 0;
    
    /**
     * @brief Reset optimizer state
     */
    virtual void reset_state() = 0;
};

} // namespace cppvision

#endif // CPPVISION_OPTIMIZERS_OPTIMIZER_H
