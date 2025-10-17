#ifndef CPPVISION_TRAINING_LOSS_H
#define CPPVISION_TRAINING_LOSS_H

#include "../core/tensor.h"
#include <vector>
#include <string>

namespace cppvision {

/**
 * @brief Abstract base class for loss functions
 * 
 * The Loss class defines the interface for all loss functions.
 * It provides methods for computing loss values and gradients.
 */
class Loss {
public:
    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~Loss() = default;

    /**
     * @brief Compute loss value
     * @param predictions Model predictions
     * @param targets Target values
     * @return Loss value
     */
    virtual double compute_loss(const Tensor& predictions, const Tensor& targets) = 0;
    
    /**
     * @brief Compute loss gradients
     * @param predictions Model predictions
     * @param targets Target values
     * @return Gradient tensor
     */
    virtual Tensor compute_gradients(const Tensor& predictions, const Tensor& targets) = 0;
    
    /**
     * @brief Get loss function name
     * @return Loss function name
     */
    virtual std::string get_name() const = 0;
};

/**
 * @brief Cross-entropy loss for classification
 * 
 * Computes cross-entropy loss between predicted probabilities and target labels.
 * Used for multi-class classification tasks.
 */
class CrossEntropyLoss : public Loss {
private:
    double epsilon_;               // Small value to avoid log(0)
    std::string name_;            // Loss function name

public:
    /**
     * @brief Constructor for cross-entropy loss
     * @param epsilon Small value to avoid numerical issues
     * @param name Loss function name
     */
    explicit CrossEntropyLoss(double epsilon = 1e-15, const std::string& name = "CrossEntropyLoss");

    /**
     * @brief Destructor
     */
    ~CrossEntropyLoss() override = default;

    /**
     * @brief Compute cross-entropy loss
     * @param predictions Predicted probabilities [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     * @return Cross-entropy loss value
     */
    double compute_loss(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute cross-entropy gradients
     * @param predictions Predicted probabilities [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     * @return Gradient tensor
     */
    Tensor compute_gradients(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Get loss function name
     * @return Loss function name
     */
    std::string get_name() const override { return name_; }

private:
    /**
     * @brief Apply softmax to predictions
     * @param predictions Raw predictions
     * @return Softmax probabilities
     */
    Tensor softmax(const Tensor& predictions) const;
    
    /**
     * @brief Clip values to avoid numerical issues
     * @param tensor Input tensor
     * @param min_val Minimum value
     * @param max_val Maximum value
     * @return Clipped tensor
     */
    Tensor clip(const Tensor& tensor, double min_val, double max_val) const;
};

/**
 * @brief Mean Squared Error loss for regression
 * 
 * Computes mean squared error between predictions and targets.
 * Used for regression tasks.
 */
class MSELoss : public Loss {
private:
    std::string name_;            // Loss function name

public:
    /**
     * @brief Constructor for MSE loss
     * @param name Loss function name
     */
    explicit MSELoss(const std::string& name = "MSELoss");

    /**
     * @brief Destructor
     */
    ~MSELoss() override = default;

    /**
     * @brief Compute MSE loss
     * @param predictions Model predictions
     * @param targets Target values
     * @return MSE loss value
     */
    double compute_loss(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute MSE gradients
     * @param predictions Model predictions
     * @param targets Target values
     * @return Gradient tensor
     */
    Tensor compute_gradients(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Get loss function name
     * @return Loss function name
     */
    std::string get_name() const override { return name_; }
};

/**
 * @brief Binary Cross-Entropy loss for binary classification
 * 
 * Computes binary cross-entropy loss between predictions and targets.
 * Used for binary classification tasks.
 */
class BCELoss : public Loss {
private:
    double epsilon_;               // Small value to avoid log(0)
    std::string name_;            // Loss function name

public:
    /**
     * @brief Constructor for BCE loss
     * @param epsilon Small value to avoid numerical issues
     * @param name Loss function name
     */
    explicit BCELoss(double epsilon = 1e-15, const std::string& name = "BCELoss");

    /**
     * @brief Destructor
     */
    ~BCELoss() override = default;

    /**
     * @brief Compute BCE loss
     * @param predictions Predicted probabilities [batch_size]
     * @param targets Target labels [batch_size]
     * @return BCE loss value
     */
    double compute_loss(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute BCE gradients
     * @param predictions Predicted probabilities [batch_size]
     * @param targets Target labels [batch_size]
     * @return Gradient tensor
     */
    Tensor compute_gradients(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Get loss function name
     * @return Loss function name
     */
    std::string get_name() const override { return name_; }

private:
    /**
     * @brief Apply sigmoid to predictions
     * @param predictions Raw predictions
     * @return Sigmoid probabilities
     */
    Tensor sigmoid(const Tensor& predictions) const;
};

/**
 * @brief Categorical Cross-Entropy loss with label smoothing
 * 
 * Computes cross-entropy loss with label smoothing regularization.
 * Helps prevent overconfidence in predictions.
 */
class LabelSmoothingCrossEntropyLoss : public Loss {
private:
    double smoothing_;             // Label smoothing factor
    double epsilon_;               // Small value to avoid log(0)
    std::string name_;            // Loss function name

public:
    /**
     * @brief Constructor for label smoothing cross-entropy loss
     * @param smoothing Label smoothing factor (0.0 to 1.0)
     * @param epsilon Small value to avoid numerical issues
     * @param name Loss function name
     */
    explicit LabelSmoothingCrossEntropyLoss(double smoothing = 0.1, double epsilon = 1e-15,
                                           const std::string& name = "LabelSmoothingCrossEntropyLoss");

    /**
     * @brief Destructor
     */
    ~LabelSmoothingCrossEntropyLoss() override = default;

    /**
     * @brief Compute label smoothing cross-entropy loss
     * @param predictions Predicted probabilities [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     * @return Loss value
     */
    double compute_loss(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute label smoothing cross-entropy gradients
     * @param predictions Predicted probabilities [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     * @return Gradient tensor
     */
    Tensor compute_gradients(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Get loss function name
     * @return Loss function name
     */
    std::string get_name() const override { return name_; }

private:
    /**
     * @brief Apply softmax to predictions
     * @param predictions Raw predictions
     * @return Softmax probabilities
     */
    Tensor softmax(const Tensor& predictions) const;
    
    /**
     * @brief Apply label smoothing to targets
     * @param targets Original targets
     * @return Smoothed targets
     */
    Tensor smooth_labels(const Tensor& targets) const;
};

} // namespace cppvision

#endif // CPPVISION_TRAINING_LOSS_H
