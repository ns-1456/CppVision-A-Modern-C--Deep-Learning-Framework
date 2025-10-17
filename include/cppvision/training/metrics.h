#ifndef CPPVISION_TRAINING_METRICS_H
#define CPPVISION_TRAINING_METRICS_H

#include "../core/tensor.h"
#include <vector>
#include <string>
#include <map>

namespace cppvision {

/**
 * @brief Abstract base class for evaluation metrics
 * 
 * The Metric class defines the interface for all evaluation metrics.
 * It provides methods for computing metric values and updating state.
 */
class Metric {
public:
    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~Metric() = default;

    /**
     * @brief Update metric with new predictions and targets
     * @param predictions Model predictions
     * @param targets Target values
     */
    virtual void update(const Tensor& predictions, const Tensor& targets) = 0;
    
    /**
     * @brief Compute current metric value
     * @return Current metric value
     */
    virtual double compute() = 0;
    
    /**
     * @brief Reset metric state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Get metric name
     * @return Metric name
     */
    virtual std::string get_name() const = 0;
};

/**
 * @brief Accuracy metric for classification
 * 
 * Computes the accuracy of predictions compared to targets.
 * For multi-class classification, uses argmax to determine predicted class.
 */
class Accuracy : public Metric {
private:
    int correct_predictions_;      // Number of correct predictions
    int total_predictions_;       // Total number of predictions
    std::string name_;            // Metric name

public:
    /**
     * @brief Constructor for accuracy metric
     * @param name Metric name
     */
    explicit Accuracy(const std::string& name = "Accuracy");

    /**
     * @brief Destructor
     */
    ~Accuracy() override = default;

    /**
     * @brief Update accuracy with new predictions and targets
     * @param predictions Model predictions [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     */
    void update(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute current accuracy
     * @return Accuracy value (0.0 to 1.0)
     */
    double compute() override;
    
    /**
     * @brief Reset accuracy state
     */
    void reset() override;
    
    /**
     * @brief Get metric name
     * @return Metric name
     */
    std::string get_name() const override { return name_; }

private:
    /**
     * @brief Get predicted class from probabilities
     * @param predictions Prediction probabilities
     * @return Predicted class indices
     */
    std::vector<int> get_predicted_classes(const Tensor& predictions) const;
    
    /**
     * @brief Get target class from one-hot encoding
     * @param targets Target labels (one-hot encoded)
     * @return Target class indices
     */
    std::vector<int> get_target_classes(const Tensor& targets) const;
};

/**
 * @brief Precision metric for classification
 * 
 * Computes precision for each class and overall precision.
 * Precision = True Positives / (True Positives + False Positives)
 */
class Precision : public Metric {
private:
    std::vector<int> true_positives_;     // True positives per class
    std::vector<int> false_positives_;   // False positives per class
    int num_classes_;                     // Number of classes
    std::string name_;                    // Metric name

public:
    /**
     * @brief Constructor for precision metric
     * @param num_classes Number of classes
     * @param name Metric name
     */
    explicit Precision(int num_classes, const std::string& name = "Precision");

    /**
     * @brief Destructor
     */
    ~Precision() override = default;

    /**
     * @brief Update precision with new predictions and targets
     * @param predictions Model predictions [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     */
    void update(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute current precision (macro-averaged)
     * @return Precision value (0.0 to 1.0)
     */
    double compute() override;
    
    /**
     * @brief Reset precision state
     */
    void reset() override;
    
    /**
     * @brief Get metric name
     * @return Metric name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get precision for each class
     * @return Vector of precision values per class
     */
    std::vector<double> get_per_class_precision() const;
};

/**
 * @brief Recall metric for classification
 * 
 * Computes recall for each class and overall recall.
 * Recall = True Positives / (True Positives + False Negatives)
 */
class Recall : public Metric {
private:
    std::vector<int> true_positives_;     // True positives per class
    std::vector<int> false_negatives_;    // False negatives per class
    int num_classes_;                     // Number of classes
    std::string name_;                    // Metric name

public:
    /**
     * @brief Constructor for recall metric
     * @param num_classes Number of classes
     * @param name Metric name
     */
    explicit Recall(int num_classes, const std::string& name = "Recall");

    /**
     * @brief Destructor
     */
    ~Recall() override = default;

    /**
     * @brief Update recall with new predictions and targets
     * @param predictions Model predictions [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     */
    void update(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute current recall (macro-averaged)
     * @return Recall value (0.0 to 1.0)
     */
    double compute() override;
    
    /**
     * @brief Reset recall state
     */
    void reset() override;
    
    /**
     * @brief Get metric name
     * @return Metric name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get recall for each class
     * @return Vector of recall values per class
     */
    std::vector<double> get_per_class_recall() const;
};

/**
 * @brief F1-Score metric for classification
 * 
 * Computes F1-score for each class and overall F1-score.
 * F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
 */
class F1Score : public Metric {
private:
    std::unique_ptr<Precision> precision_;   // Precision metric
    std::unique_ptr<Recall> recall_;          // Recall metric
    std::string name_;                        // Metric name

public:
    /**
     * @brief Constructor for F1-score metric
     * @param num_classes Number of classes
     * @param name Metric name
     */
    explicit F1Score(int num_classes, const std::string& name = "F1Score");

    /**
     * @brief Destructor
     */
    ~F1Score() override = default;

    /**
     * @brief Update F1-score with new predictions and targets
     * @param predictions Model predictions [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     */
    void update(const Tensor& predictions, const Tensor& targets) override;
    
    /**
     * @brief Compute current F1-score (macro-averaged)
     * @return F1-score value (0.0 to 1.0)
     */
    double compute() override;
    
    /**
     * @brief Reset F1-score state
     */
    void reset() override;
    
    /**
     * @brief Get metric name
     * @return Metric name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get F1-score for each class
     * @return Vector of F1-score values per class
     */
    std::vector<double> get_per_class_f1() const;
};

/**
 * @brief Confusion matrix for classification
 * 
 * Computes and stores confusion matrix for multi-class classification.
 * Provides detailed analysis of prediction performance.
 */
class ConfusionMatrix {
private:
    std::vector<std::vector<int>> matrix_;    // Confusion matrix
    int num_classes_;                         // Number of classes
    std::vector<std::string> class_names_;    // Class names

public:
    /**
     * @brief Constructor for confusion matrix
     * @param num_classes Number of classes
     * @param class_names Optional class names
     */
    explicit ConfusionMatrix(int num_classes, 
                           const std::vector<std::string>& class_names = {});

    /**
     * @brief Destructor
     */
    ~ConfusionMatrix() = default;

    /**
     * @brief Update confusion matrix with new predictions and targets
     * @param predictions Model predictions [batch_size, num_classes]
     * @param targets Target labels (one-hot encoded) [batch_size, num_classes]
     */
    void update(const Tensor& predictions, const Tensor& targets);
    
    /**
     * @brief Reset confusion matrix
     */
    void reset();
    
    /**
     * @brief Get confusion matrix
     * @return Confusion matrix
     */
    const std::vector<std::vector<int>>& get_matrix() const { return matrix_; }
    
    /**
     * @brief Print confusion matrix
     */
    void print() const;
    
    /**
     * @brief Get confusion matrix as string
     * @return String representation of confusion matrix
     */
    std::string to_string() const;
    
    /**
     * @brief Get precision for each class
     * @return Vector of precision values per class
     */
    std::vector<double> get_precision_per_class() const;
    
    /**
     * @brief Get recall for each class
     * @return Vector of recall values per class
     */
    std::vector<double> get_recall_per_class() const;
    
    /**
     * @brief Get F1-score for each class
     * @return Vector of F1-score values per class
     */
    std::vector<double> get_f1_per_class() const;

private:
    /**
     * @brief Get predicted class from probabilities
     * @param predictions Prediction probabilities
     * @return Predicted class indices
     */
    std::vector<int> get_predicted_classes(const Tensor& predictions) const;
    
    /**
     * @brief Get target class from one-hot encoding
     * @param targets Target labels (one-hot encoded)
     * @return Target class indices
     */
    std::vector<int> get_target_classes(const Tensor& targets) const;
};

/**
 * @brief Metric collection for managing multiple metrics
 * 
 * Provides a convenient way to manage and compute multiple metrics together.
 */
class MetricCollection {
private:
    std::vector<std::unique_ptr<Metric>> metrics_;     // Collection of metrics
    std::map<std::string, double> metric_values_;      // Current metric values

public:
    /**
     * @brief Default constructor
     */
    MetricCollection() = default;

    /**
     * @brief Destructor
     */
    ~MetricCollection() = default;

    /**
     * @brief Add a metric to the collection
     * @param metric Unique pointer to metric
     */
    void add_metric(std::unique_ptr<Metric> metric);
    
    /**
     * @brief Update all metrics with new predictions and targets
     * @param predictions Model predictions
     * @param targets Target values
     */
    void update(const Tensor& predictions, const Tensor& targets);
    
    /**
     * @brief Compute all metric values
     * @return Map of metric names to values
     */
    std::map<std::string, double> compute();
    
    /**
     * @brief Reset all metrics
     */
    void reset();
    
    /**
     * @brief Get number of metrics
     * @return Number of metrics
     */
    int size() const { return static_cast<int>(metrics_.size()); }
    
    /**
     * @brief Check if collection is empty
     * @return True if empty
     */
    bool empty() const { return metrics_.empty(); }
    
    /**
     * @brief Print all metric values
     */
    void print() const;
    
    /**
     * @brief Get metric values as string
     * @return String representation of metric values
     */
    std::string to_string() const;
};

} // namespace cppvision

#endif // CPPVISION_TRAINING_METRICS_H
