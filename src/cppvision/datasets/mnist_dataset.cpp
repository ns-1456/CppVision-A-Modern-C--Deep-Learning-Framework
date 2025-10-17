#include "cppvision/datasets/mnist_dataset.h"
#include "cppvision/core/tensor.h"
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstdint>

namespace cppvision {

// Constructor
MNISTDataset::MNISTDataset(const std::string& images_path, const std::string& labels_path,
                           const std::string& name)
    : current_index_(0), name_(name), augmentation_enabled_(false), rng_(std::random_device{}()) {
    // Load MNIST data
    load_mnist_data(images_path, labels_path);
    
    // Initialize indices
    indices_.resize(images_.size());
    std::iota(indices_.begin(), indices_.end(), 0);
    
    std::cout << "Loaded MNIST dataset: " << images_.size() << " samples" << std::endl;
}

// Dataset interface methods
std::pair<Tensor, Tensor> MNISTDataset::get_batch(int batch_size) {
    int actual_batch_size = std::min(batch_size, static_cast<int>(images_.size() - current_index_));
    
    // Create batch tensors
    std::vector<int> image_shape = {actual_batch_size, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE};
    std::vector<int> label_shape = {actual_batch_size, NUM_CLASSES};
    
    Tensor batch_images = Tensor::zeros(image_shape, false);
    Tensor batch_labels = Tensor::zeros(label_shape, false);
    
    // Fill batch with data
    for (int i = 0; i < actual_batch_size; ++i) {
        int data_idx = indices_[current_index_ + i];
        
        // Copy image data
        for (int h = 0; h < IMAGE_SIZE; ++h) {
            for (int w = 0; w < IMAGE_SIZE; ++w) {
                int img_idx = h * IMAGE_SIZE + w;
                int batch_idx = i * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE + img_idx;
                batch_images[batch_idx] = images_[data_idx][img_idx];
            }
        }
        
        // Copy label data (one-hot encoded)
        for (int c = 0; c < NUM_CLASSES; ++c) {
            int label_idx = i * NUM_CLASSES + c;
            batch_labels[label_idx] = labels_[data_idx][c];
        }
    }
    
    current_index_ += actual_batch_size;
    return std::make_pair(std::move(batch_images), std::move(batch_labels));
}

int MNISTDataset::size() const {
    return static_cast<int>(images_.size());
}

void MNISTDataset::shuffle() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
    current_index_ = 0;
}

// Additional required methods from Dataset interface
std::pair<Tensor, Tensor> MNISTDataset::get_sample(int index) {
    if (index < 0 || index >= static_cast<int>(images_.size())) {
        throw std::out_of_range("Index out of range");
    }
    
    return std::make_pair(images_[index], labels_[index]);
}

bool MNISTDataset::empty() const {
    return images_.empty();
}

void MNISTDataset::reset() {
    current_index_ = 0;
}

void MNISTDataset::set_seed(int seed) {
    rng_.seed(seed);
}

std::string MNISTDataset::get_name() const {
    return name_;
}

std::vector<int> MNISTDataset::get_feature_shape() const {
    return {NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE};
}

int MNISTDataset::get_num_classes() const {
    return NUM_CLASSES;
}

std::vector<std::string> MNISTDataset::get_class_names() const {
    return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
}

void MNISTDataset::enable_augmentation(bool enable) {
    augmentation_enabled_ = enable;
}

bool MNISTDataset::is_augmentation_enabled() const {
    return augmentation_enabled_;
}

// Private helper methods
void MNISTDataset::load_mnist_data(const std::string& images_path, const std::string& labels_path) {
    // Load images
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file.is_open()) {
        throw std::runtime_error("Cannot open MNIST images file: " + images_path);
    }
    
    // Read header
    uint32_t magic_number, num_images, num_rows, num_cols;
    images_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    images_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    images_file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    images_file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    
    // Convert from big-endian
    magic_number = reverse_int(magic_number);
    num_images = reverse_int(num_images);
    num_rows = reverse_int(num_rows);
    num_cols = reverse_int(num_cols);
    
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST images file magic number");
    }
    
    // Load images
    images_.resize(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<double> image_data(num_rows * num_cols);
        images_file.read(reinterpret_cast<char*>(image_data.data()), num_rows * num_cols);
        
        // Normalize to [0, 1]
        for (double& pixel : image_data) {
            pixel = pixel / 255.0;
        }
        
        images_[i] = Tensor(image_data, {NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE}, false);
    }
    
    images_file.close();
    
    // Load labels
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file.is_open()) {
        throw std::runtime_error("Cannot open MNIST labels file: " + labels_path);
    }
    
    // Read header
    uint32_t label_magic, num_labels;
    labels_file.read(reinterpret_cast<char*>(&label_magic), sizeof(label_magic));
    labels_file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    
    // Convert from big-endian
    label_magic = reverse_int(label_magic);
    num_labels = reverse_int(num_labels);
    
    if (label_magic != 2049) {
        throw std::runtime_error("Invalid MNIST labels file magic number");
    }
    
    // Load labels
    labels_.resize(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t label;
        labels_file.read(reinterpret_cast<char*>(&label), sizeof(label));
        
        // Convert to one-hot encoding
        std::vector<double> one_hot(NUM_CLASSES, 0.0);
        one_hot[label] = 1.0;
        
        labels_[i] = Tensor(one_hot, {NUM_CLASSES}, false);
    }
    
    labels_file.close();
}

uint32_t MNISTDataset::reverse_int(uint32_t i) {
    uint8_t c1 = (i >> 24) & 0xFF;
    uint8_t c2 = (i >> 16) & 0xFF;
    uint8_t c3 = (i >> 8) & 0xFF;
    uint8_t c4 = i & 0xFF;
    
    return (static_cast<uint32_t>(c4) << 24) | 
           (static_cast<uint32_t>(c3) << 16) | 
           (static_cast<uint32_t>(c2) << 8) | 
           static_cast<uint32_t>(c1);
}

} // namespace cppvision