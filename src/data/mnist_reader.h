#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <vector>
#include <string>

// The reverseInt function is a utility that doesn't need to be
// exposed to the rest of the project, so we often leave it
// out of the header. But if you needed it, you would put it here.

// Function to read the MNIST Label File
// Returns a vector of single-byte labels (0-9)
std::vector<uint8_t> read_mnist_labels(const std::string& full_path);

// Function to read the MNIST Image File
// Returns a vector of vectors (list of images, where each image is a list of 784 pixels)
std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& full_path);

#endif // MNIST_READER_H