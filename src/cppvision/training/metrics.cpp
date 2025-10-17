#include "cppvision/training/metrics.h"
#include "cppvision/core/tensor.h"
#include <algorithm>
#include <cmath>

namespace cppvision {

// Accuracy implementation
Accuracy::Accuracy(const std::string& name) : correct_predictions_(0), total_predictions_(0), name_(name) {
    // Constructor
}

void Accuracy::update(const Tensor& predictions, const Tensor& targets) {
    // Update accuracy with new predictions and targets
    // Algorithm:
    //   1. Find predicted class (argmax of predictions)
    //   2. Find true class (argmax of targets)
    //   3. Count correct predictions
    
    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    
    for (int i = 0; i < batch_size; ++i) {
        // Find predicted class
        int pred_class = 0;
        double max_pred = predictions[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (predictions[idx] > max_pred) {
                max_pred = predictions[idx];
                pred_class = j;
            }
        }
        
        // Find true class
        int true_class = 0;
        double max_target = targets[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (targets[idx] > max_target) {
                max_target = targets[idx];
                true_class = j;
            }
        }
        
        // Count correct predictions
        if (pred_class == true_class) {
            correct_predictions_++;
        }
        total_predictions_++;
    }
}

double Accuracy::compute() {
    // Compute current accuracy
    if (total_predictions_ == 0) return 0.0;
    return static_cast<double>(correct_predictions_) / total_predictions_;
}

void Accuracy::reset() {
    // Reset accuracy state
    correct_predictions_ = 0;
    total_predictions_ = 0;
}

// Precision implementation
Precision::Precision(int num_classes, const std::string& name) 
    : num_classes_(num_classes), name_(name) {
    // Initialize confusion matrix
    true_positives_.resize(num_classes_, 0);
    false_positives_.resize(num_classes_, 0);
}

void Precision::update(const Tensor& predictions, const Tensor& targets) {
    // Update precision with new predictions and targets
    // Algorithm:
    //   1. Find predicted and true classes
    //   2. Update confusion matrix
    
    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    
    for (int i = 0; i < batch_size; ++i) {
        // Find predicted class
        int pred_class = 0;
        double max_pred = predictions[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (predictions[idx] > max_pred) {
                max_pred = predictions[idx];
                pred_class = j;
            }
        }
        
        // Find true class
        int true_class = 0;
        double max_target = targets[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (targets[idx] > max_target) {
                max_target = targets[idx];
                true_class = j;
            }
        }
        
        // Update confusion matrix
        if (pred_class == true_class) {
            true_positives_[pred_class]++;
        } else {
            false_positives_[pred_class]++;
        }
    }
}

double Precision::compute() {
    // Compute average precision across all classes
    double total_precision = 0.0;
    int valid_classes = 0;
    
    for (int i = 0; i < num_classes_; ++i) {
        int tp = true_positives_[i];
        int fp = false_positives_[i];
        
        if (tp + fp > 0) {
            total_precision += static_cast<double>(tp) / (tp + fp);
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? total_precision / valid_classes : 0.0;
}

void Precision::reset() {
    // Reset precision state
    std::fill(true_positives_.begin(), true_positives_.end(), 0);
    std::fill(false_positives_.begin(), false_positives_.end(), 0);
}

// Recall implementation
Recall::Recall(int num_classes, const std::string& name) 
    : num_classes_(num_classes), name_(name) {
    // Initialize confusion matrix
    true_positives_.resize(num_classes_, 0);
    false_negatives_.resize(num_classes_, 0);
}

void Recall::update(const Tensor& predictions, const Tensor& targets) {
    // Update recall with new predictions and targets
    // Algorithm:
    //   1. Find predicted and true classes
    //   2. Update confusion matrix
    
    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    
    for (int i = 0; i < batch_size; ++i) {
        // Find predicted class
        int pred_class = 0;
        double max_pred = predictions[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (predictions[idx] > max_pred) {
                max_pred = predictions[idx];
                pred_class = j;
            }
        }
        
        // Find true class
        int true_class = 0;
        double max_target = targets[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (targets[idx] > max_target) {
                max_target = targets[idx];
                true_class = j;
            }
        }
        
        // Update confusion matrix
        if (pred_class == true_class) {
            true_positives_[true_class]++;
        } else {
            false_negatives_[true_class]++;
        }
    }
}

double Recall::compute() {
    // Compute average recall across all classes
    double total_recall = 0.0;
    int valid_classes = 0;
    
    for (int i = 0; i < num_classes_; ++i) {
        int tp = true_positives_[i];
        int fn = false_negatives_[i];
        
        if (tp + fn > 0) {
            total_recall += static_cast<double>(tp) / (tp + fn);
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? total_recall / valid_classes : 0.0;
}

void Recall::reset() {
    // Reset recall state
    std::fill(true_positives_.begin(), true_positives_.end(), 0);
    std::fill(false_negatives_.begin(), false_negatives_.end(), 0);
}

// F1Score implementation
F1Score::F1Score(int num_classes, const std::string& name) 
    : name_(name) {
    // Initialize precision and recall metrics
    precision_ = std::make_unique<Precision>(num_classes, "Precision");
    recall_ = std::make_unique<Recall>(num_classes, "Recall");
}

void F1Score::update(const Tensor& predictions, const Tensor& targets) {
    // Update F1 score with new predictions and targets
    // Algorithm:
    //   1. Update precision and recall metrics
    precision_->update(predictions, targets);
    recall_->update(predictions, targets);
}

double F1Score::compute() {
    // Compute F1 score using precision and recall
    // Algorithm:
    //   1. Get precision and recall values
    double precision_val = precision_->compute();
    double recall_val = recall_->compute();
    
    //   2. Compute F1 score: 2 * (precision * recall) / (precision + recall)
    if (precision_val + recall_val > 0) {
        return 2.0 * precision_val * recall_val / (precision_val + recall_val);
    }
    
    return 0.0;
}

void F1Score::reset() {
    // Reset F1 score state
    precision_->reset();
    recall_->reset();
}

// MetricCollection implementation
void MetricCollection::add_metric(std::unique_ptr<Metric> metric) {
    // Add metric to collection
    metrics_.push_back(std::move(metric));
}

void MetricCollection::update(const Tensor& predictions, const Tensor& targets) {
    // Update all metrics
    for (auto& metric : metrics_) {
        metric->update(predictions, targets);
    }
}

std::map<std::string, double> MetricCollection::compute() {
    // Compute all metrics
    std::map<std::string, double> results;
    for (auto& metric : metrics_) {
        results[metric->get_name()] = metric->compute();
    }
    return results;
}

void MetricCollection::reset() {
    // Reset all metrics
    for (auto& metric : metrics_) {
        metric->reset();
    }
}

} // namespace cppvision