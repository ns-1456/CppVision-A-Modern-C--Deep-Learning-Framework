#ifndef CPPVISION_DATASETS_MNIST_DATASET_H
#define CPPVISION_DATASETS_MNIST_DATASET_H

#include "dataset.h"
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <random>
#include <cstdint>

namespace cppvision {

/**
 * @brief MNIST dataset implementation
 * 
 * Loads and provides access to the MNIST handwritten digit dataset.
 * Supports both training and test sets, with optional data augmentation.
 */
class MNISTDataset : public Dataset {
private:
    std::vector<Tensor> images_;          // Image tensors
    std::vector<Tensor> labels_;           // Label tensors (one-hot encoded)
    std::vector<int> indices_;            // Shuffled indices
    int current_index_;                   // Current position in dataset
    std::string name_;                    // Dataset name
    bool augmentation_enabled_;           // Whether augmentation is enabled
    std::mt19937 rng_;                    // Random number generator
    
    // MNIST-specific parameters
    static constexpr int IMAGE_SIZE = 28;     // MNIST image size
    static constexpr int NUM_CLASSES = 10;    // Number of digit classes
    static constexpr int NUM_CHANNELS = 1;    // Grayscale images

public:
    /**
     * @brief Constructor for MNIST dataset
     * @param images_path Path to MNIST images file
     * @param labels_path Path to MNIST labels file
     * @param name Dataset name (e.g., "MNIST-Train", "MNIST-Test")
     */
    MNISTDataset(const std::string& images_path, const std::string& labels_path,
                 const std::string& name = "MNIST");

    /**
     * @brief Destructor
     */
    ~MNISTDataset() override = default;

    // Data access
    /**
     * @brief Get a batch of data
     * @param batch_size Number of samples in the batch
     * @return Pair of (images, labels) tensors
     */
    std::pair<Tensor, Tensor> get_batch(int batch_size) override;
    
    /**
     * @brief Get a single sample
     * @param index Sample index
     * @return Pair of (image, label) tensors
     */
    std::pair<Tensor, Tensor> get_sample(int index) override;
    
    /**
     * @brief Get total number of samples
     * @return Dataset size
     */
    int size() const override { return static_cast<int>(images_.size()); }
    
    /**
     * @brief Check if dataset is empty
     * @return True if dataset is empty
     */
    bool empty() const override { return images_.empty(); }

    // Data manipulation
    /**
     * @brief Shuffle the dataset
     */
    void shuffle() override;
    
    /**
     * @brief Reset dataset to initial state
     */
    void reset() override;
    
    /**
     * @brief Set random seed for reproducibility
     * @param seed Random seed
     */
    void set_seed(int seed) override;

    // Dataset information
    /**
     * @brief Get dataset name
     * @return Dataset name
     */
    std::string get_name() const override { return name_; }
    
    /**
     * @brief Get feature shape
     * @return Shape of image tensors [height, width, channels]
     */
    std::vector<int> get_feature_shape() const override;
    
    /**
     * @brief Get number of classes
     * @return Number of classes (10 for MNIST)
     */
    int get_num_classes() const override { return NUM_CLASSES; }
    
    /**
     * @brief Get class names
     * @return Vector of class names ("0", "1", ..., "9")
     */
    std::vector<std::string> get_class_names() const override;

    // Data augmentation
    /**
     * @brief Enable data augmentation
     * @param enable True to enable augmentation
     */
    void enable_augmentation(bool enable) override;
    
    /**
     * @brief Check if augmentation is enabled
     * @return True if augmentation is enabled
     */
    bool is_augmentation_enabled() const override { return augmentation_enabled_; }

    // Iterator support - REMOVED (not in base class)
    // auto begin() override;
    // auto end() override;

    // MNIST-specific methods
    /**
     * @brief Load MNIST data from files
     * @param images_path Path to images file
     * @param labels_path Path to labels file
     */
    void load_data(const std::string& images_path, const std::string& labels_path);
    
    /**
     * @brief Convert raw MNIST data to tensors
     * @param raw_images Raw image data
     * @param raw_labels Raw label data
     */
    void convert_to_tensors(const std::vector<std::vector<uint8_t>>& raw_images,
                          const std::vector<uint8_t>& raw_labels);
    
    /**
     * @brief Normalize images to [0, 1] range
     * @param images Image tensors
     * @return Normalized image tensors
     */
    std::vector<Tensor> normalize_images(const std::vector<Tensor>& images) const;
    
    /**
     * @brief Convert labels to one-hot encoding
     * @param labels Label tensors
     * @return One-hot encoded label tensors
     */
    std::vector<Tensor> one_hot_encode_labels(const std::vector<Tensor>& labels) const;

    // Data augmentation methods
    /**
     * @brief Apply random rotation to image
     * @param image Input image tensor
     * @param max_angle Maximum rotation angle in degrees
     * @return Augmented image tensor
     */
    Tensor random_rotation(const Tensor& image, double max_angle = 15.0) const;
    
    /**
     * @brief Apply random translation to image
     * @param image Input image tensor
     * @param max_translation Maximum translation in pixels
     * @return Augmented image tensor
     */
    Tensor random_translation(const Tensor& image, int max_translation = 2) const;
    
    /**
     * @brief Apply random noise to image
     * @param image Input image tensor
     * @param noise_std Standard deviation of noise
     * @return Augmented image tensor
     */
    Tensor random_noise(const Tensor& image, double noise_std = 0.1) const;
    
    /**
     * @brief Apply random brightness adjustment
     * @param image Input image tensor
     * @param brightness_range Brightness adjustment range
     * @return Augmented image tensor
     */
    Tensor random_brightness(const Tensor& image, double brightness_range = 0.2) const;

private:
    /**
     * @brief Read MNIST images from file
     * @param filename Path to images file
     * @return Vector of image vectors
     */
    std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& filename) const;
    
    /**
     * @brief Read MNIST labels from file
     * @param filename Path to labels file
     * @return Vector of labels
     */
    std::vector<uint8_t> read_mnist_labels(const std::string& filename) const;
    
    /**
     * @brief Convert uint8_t vector to double tensor
     * @param data Raw data vector
     * @param shape Tensor shape
     * @return Double tensor
     */
    Tensor vector_to_tensor(const std::vector<uint8_t>& data, const std::vector<int>& shape) const;
    
    /**
     * @brief Apply augmentation to image if enabled
     * @param image Input image tensor
     * @return Augmented image tensor
     */
    Tensor apply_augmentation(const Tensor& image) const;
    
    /**
     * @brief Initialize shuffled indices
     */
    void initialize_indices();
};

} // namespace cppvision

#endif // CPPVISION_DATASETS_MNIST_DATASET_H
