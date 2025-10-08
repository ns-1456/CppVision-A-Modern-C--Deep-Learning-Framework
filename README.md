# C++ MNIST Classifier

A Convolutional Neural Network (CNN) implementation in C++ for classifying handwritten digits from the MNIST dataset.

## Overview

This project implements a complete CNN pipeline for MNIST digit classification, including:
- Data loading and preprocessing
- Neural network layers (Convolutional, Dense, ReLU, Softmax, MaxPooling)
- Training with backpropagation
- Model evaluation and metrics

## Project Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── layers/         # Neural network layer implementations
│   ├── training/       # Training algorithms and loss functions
│   └── main.cpp        # Main application entry point
├── data/               # MNIST dataset files
├── include/            # Header files
├── docs/               # Documentation
└── CMakeLists.txt      # Build configuration
```

## Prerequisites

- C++17 compatible compiler (GCC, Clang, or MSVC)
- CMake 3.15 or higher
- MNIST dataset files (included in `data/` directory)

## Building the Project

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cpp-mnist-classifier.git
cd cpp-mnist-classifier
```

2. Create a build directory:
```bash
mkdir build
cd build
```

3. Configure and build:
```bash
cmake ..
make
```

## Usage

Run the trained model:
```bash
./mnist_cnn
```

## Dataset

The MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images of handwritten digits (0-9)

Dataset files are included in the `data/` directory:
- `train-images-idx3-ubyte` - Training images
- `train-labels-idx1-ubyte` - Training labels
- `t10k-images-idx3-ubyte` - Test images
- `t10k-labels-idx1-ubyte` - Test labels

## Features

- **Convolutional Layers**: 2D convolution operations
- **Pooling Layers**: Max pooling for dimensionality reduction
- **Activation Functions**: ReLU and Softmax
- **Dense Layers**: Fully connected layers
- **Training**: Backpropagation with gradient descent
- **Metrics**: Accuracy, loss calculation, and performance evaluation

## Architecture

The CNN architecture includes:
1. Convolutional layer with ReLU activation
2. Max pooling layer
3. Convolutional layer with ReLU activation
4. Max pooling layer
5. Dense (fully connected) layer
6. Softmax output layer

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MNIST dataset by Yann LeCun
- C++ implementation inspired by modern deep learning frameworks
