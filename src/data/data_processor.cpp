#include "data_processor.h"
#include <iostream>

// Converts pixel values from [0, 255] (uint8_t) to [0.0, 1.0] (double)
std::vector<std::vector<double>> normalize_images(const std::vector<std::vector<uint8_t>>& raw_images)
{
    // Pre-allocate the outer vector for efficiency
    std::vector<std::vector<double>> normalized_images;
    normalized_images.reserve(raw_images.size());
    
    std::cout << "   - Normalizing " << raw_images.size() << " images..." << std::endl;

    for (const auto& raw_image : raw_images) {
        // Pre-allocate the inner vector for efficiency
        std::vector<double> normalized_image;
        normalized_image.reserve(raw_image.size());
        
        for (uint8_t pixel : raw_image) {
            // **Normalization Step:** Divide by 255.0 to scale to [0.0, 1.0]
            normalized_image.push_back(static_cast<double>(pixel) / 255.0);
        }
        normalized_images.push_back(std::move(normalized_image));
    }
    return normalized_images;
}

// Converts a single digit label (e.g., 5) into a probability vector [0,0,0,0,0,1,0,0,0,0]
std::vector<std::vector<double>> one_hot_encode_labels(
    const std::vector<uint8_t>& raw_labels, int num_classes)
{
    std::vector<std::vector<double>> one_hot_labels;
    one_hot_labels.reserve(raw_labels.size());

    std::cout << "   - One-hot encoding " << raw_labels.size() << " labels..." << std::endl;

    for (uint8_t label : raw_labels) {
        // Create a vector of size 10, initialized to 0.0
        std::vector<double> encoded_label(num_classes, 0.0);
        
        // Set the index corresponding to the label (e.g., index 5 for label 5) to 1.0
        encoded_label[label] = 1.0; 
        
        one_hot_labels.push_back(std::move(encoded_label));
    }
    return one_hot_labels;
}