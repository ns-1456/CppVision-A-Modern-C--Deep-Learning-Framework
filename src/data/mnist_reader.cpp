#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint> // For uint8_t (unsigned 8-bit integer, used for labels)
#include <string>
// ---------------------------------------------------------------------------------------------------------------------
// Tool: Converts 4-byte Big-Endian integer from the file
// into Little-Endian integer for your computer.
uint32_t reverseInt(uint32_t i) {
    // Cast the input int to uint32_t for clean bitwise operations
    uint8_t c1 = (uint8_t)i;
    uint8_t c2 = (uint8_t)(i >> 8);
    uint8_t c3 = (uint8_t)(i >> 16);
    uint8_t c4 = (uint8_t)(i >> 24);

    // Recombine bytes in reverse order (c4, c3, c2, c1)
    return ((uint32_t)c1 << 24) | ((uint32_t)c2 << 16) | ((uint32_t)c3 << 8) | c4;
}
// ---------------------------------------------------------------------------------------------------------------------
// Function to read the MNIST Label File
std::vector<uint8_t> read_mnist_labels(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    //opening the stream
    if (!file.is_open()) {
        std::cerr << "Error opening label file: " << full_path << std::endl;
        return {};
    }
    // Reserving the space
    int magic_number = 0;
    int number_of_items = 0;

    // 1. Read the Magic Number (4 bytes)
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    // 2. Read the Number of Items (4 bytes)
    file.read((char*)&number_of_items, sizeof(number_of_items));
    number_of_items = reverseInt(number_of_items);

    // 3. Prepare a vector to hold the labels (60k labels)
    std::vector<uint8_t> labels(number_of_items);

    // 4. Read all the labels (1 byte each) into the vector
    // The data starts immediately after the 8-byte header.
    // (efficient: read all 60k labels (batch reading) at the same time in one step)
    file.read((char*)labels.data(), number_of_items);

    return labels;
}
// ---------------------------------------------------------------------------------------------------------------------
// Function to read the MNIST Image File
std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    //opening the stream
    if (!file.is_open()) {
        std::cerr << "Error opening image file: " << full_path << std::endl;
        return {};
    }
    // Reserving the space
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // 1. Read and reverse the entire 16-byte header
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);

    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    // --- CRUCIAL CHECK ---
    if (n_rows <= 0 || n_cols <= 0) {
        std::cerr << "Error: Image dimensions (Rows=" << n_rows
                  << ", Cols=" << n_cols << ") are invalid after reading header." << std::endl;
        return {};
    }
    // --- End of check ---

    // --- Data is ready to be read here ---

    // 2. Calculate the size of a single image (in bytes/pixels)
    int image_size = n_rows * n_cols; // 28 * 28 = 784

    // 3. Prepare the final list of lists (vector of vectors)
    std::vector<std::vector<uint8_t>> images(number_of_images);

    // 4. Loop to read ALL images (60k times)
    for (int i = 0; i < number_of_images; ++i) {

        // a. Prepare a new inner list (vector) for the current image
        images[i].resize(image_size);

        // b. Perform the fast block read for this one image (efficient again)
        // Read 784 bytes directly into the memory allocated for images[i]
        file.read((char*)images[i].data(), image_size);
    }

    return images;
}
// ---------------------------------------------------------------------------------------------------------------------