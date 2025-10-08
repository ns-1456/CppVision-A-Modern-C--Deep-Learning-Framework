#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H

#include <vector>
#include <cstdint>

// Function declaration to convert raw images to normalized doubles
std::vector<std::vector<double>> normalize_images(
    const std::vector<std::vector<uint8_t>>& raw_images);

// Function declaration for one-hot encoding labels
std::vector<std::vector<double>> one_hot_encode_labels(
    const std::vector<uint8_t>& raw_labels, int num_classes = 10);

#endif // DATA_PROCESSOR_H