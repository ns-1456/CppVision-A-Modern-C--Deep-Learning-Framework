#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "data/mnist_reader.h"
#include "data/data_processor.h" // <-- Include the new header

int main() {
    std::cout << "--- Starting MNIST CNN Classifier ---" << std::endl;
    // ... file paths defined here ...
    const std::string TRAIN_IMG_PATH = "/Users/ns/Desktop/Machine Learning Practice/C++ MINST Classifier/data/train-images-idx3-ubyte";
    const std::string TRAIN_LBL_PATH = "/Users/ns/Desktop/Machine Learning Practice/C++ MINST Classifier/data/train-labels-idx1-ubyte";
    const std::string TEST_IMG_PATH  = "/Users/ns/Desktop/Machine Learning Practice/C++ MINST Classifier/data/t10k-images-idx3-ubyte";
    const std::string TEST_LBL_PATH  = "/Users/ns/Desktop/Machine Learning Practice/C++ MINST Classifier/data/t10k-labels-idx1-ubyte";
    // --- 1. Data Loading Phase ---
    std::cout << "\n[1] Loading MNIST Data..." << std::endl;

    // Load RAW data (uint8_t)
    std::vector<std::vector<uint8_t>> raw_train_images = read_mnist_images(TRAIN_IMG_PATH);
    std::vector<uint8_t> raw_train_labels = read_mnist_labels(TRAIN_LBL_PATH);
    // ... Load raw test data ...
    std::vector<std::vector<uint8_t>> raw_test_images = read_mnist_images(TEST_IMG_PATH);
    std::vector<uint8_t> raw_test_labels = read_mnist_labels(TEST_LBL_PATH);

    // --- 2. Data Processing Phase ---
    std::cout << "\n[2] Processing Data (Normalization and One-Hot Encoding)..." << std::endl;

    // A. Normalize Images
    std::vector<std::vector<double>> train_images = normalize_images(raw_train_images);
    std::vector<std::vector<double>> test_images = normalize_images(raw_test_images);

    // B. One-Hot Encode Labels
    // The network output will be a 10-element probability vector, so labels need to match.
    std::vector<std::vector<double>> train_labels = one_hot_encode_labels(raw_train_labels);
    std::vector<std::vector<double>> test_labels = one_hot_encode_labels(raw_test_labels);

    std::cout << "-> Processed training set: " << train_images.size()
              << " images, " << train_labels.size() << " labels." << std::endl;

    // --- 3. Model Building and Training Phase (Next Step) ---
    // ... Rest of the main.cpp logic goes here, using the new 'train_images' and 'train_labels'

    return 0;
}