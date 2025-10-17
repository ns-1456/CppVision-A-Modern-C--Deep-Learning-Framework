#ifndef CPPVISION_DATASETS_DATASET_H
#define CPPVISION_DATASETS_DATASET_H

#include "../core/tensor.h"
#include <vector>
#include <string>
#include <utility>

namespace cppvision {

/**
 * @brief Abstract base class for all datasets
 * 
 * The Dataset class defines the interface that all datasets must implement.
 * It provides methods for data loading, batching, and shuffling.
 */
class Dataset {
public:
    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~Dataset() = default;

    // Data access
    /**
     * @brief Get a batch of data
     * @param batch_size Number of samples in the batch
     * @return Pair of (features, labels) tensors
     */
    virtual std::pair<Tensor, Tensor> get_batch(int batch_size) = 0;
    
    /**
     * @brief Get a single sample
     * @param index Sample index
     * @return Pair of (feature, label) tensors
     */
    virtual std::pair<Tensor, Tensor> get_sample(int index) = 0;
    
    /**
     * @brief Get total number of samples
     * @return Dataset size
     */
    virtual int size() const = 0;
    
    /**
     * @brief Check if dataset is empty
     * @return True if dataset is empty
     */
    virtual bool empty() const = 0;

    // Data manipulation
    /**
     * @brief Shuffle the dataset
     */
    virtual void shuffle() = 0;
    
    /**
     * @brief Reset dataset to initial state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Set random seed for reproducibility
     * @param seed Random seed
     */
    virtual void set_seed(int seed) = 0;

    // Dataset information
    /**
     * @brief Get dataset name
     * @return Dataset name
     */
    virtual std::string get_name() const = 0;
    
    /**
     * @brief Get feature shape
     * @return Shape of feature tensors
     */
    virtual std::vector<int> get_feature_shape() const = 0;
    
    /**
     * @brief Get number of classes
     * @return Number of classes (for classification tasks)
     */
    virtual int get_num_classes() const = 0;
    
    /**
     * @brief Get class names
     * @return Vector of class names
     */
    virtual std::vector<std::string> get_class_names() const = 0;

    // Data augmentation (optional)
    /**
     * @brief Enable data augmentation
     * @param enable True to enable augmentation
     */
    virtual void enable_augmentation(bool enable) {}
    
    /**
     * @brief Check if augmentation is enabled
     * @return True if augmentation is enabled
     */
    virtual bool is_augmentation_enabled() const { return false; }

    // Iterator support - REMOVED (virtual auto not supported in C++17)
    // virtual auto begin() = 0;
    // virtual auto end() = 0;
};

/**
 * @brief Dataset iterator for range-based loops
 */
class DatasetIterator {
private:
    Dataset* dataset_;
    int current_index_;
    int batch_size_;

public:
    /**
     * @brief Constructor
     * @param dataset Pointer to dataset
     * @param start_index Starting index
     * @param batch_size Batch size for iteration
     */
    DatasetIterator(Dataset* dataset, int start_index, int batch_size);
    
    /**
     * @brief Dereference operator
     * @return Batch of data
     */
    std::pair<Tensor, Tensor> operator*();
    
    /**
     * @brief Increment operator
     * @return Reference to iterator
     */
    DatasetIterator& operator++();
    
    /**
     * @brief Equality operator
     * @param other Other iterator
     * @return True if equal
     */
    bool operator==(const DatasetIterator& other) const;
    
    /**
     * @brief Inequality operator
     * @param other Other iterator
     * @return True if not equal
     */
    bool operator!=(const DatasetIterator& other) const;
};

} // namespace cppvision

#endif // CPPVISION_DATASETS_DATASET_H
