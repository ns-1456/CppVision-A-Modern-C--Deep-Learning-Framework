# CppVision Framework

A comprehensive C++ deep learning framework for computer vision tasks, designed for educational purposes and research experimentation.

## 🚀 Overview

CppVision is a modular, PyTorch-inspired deep learning framework implemented in C++17. It provides a complete toolkit for building, training, and evaluating neural networks, with a focus on computer vision applications.

## ✨ Features

### Core Components
- **Tensor Operations**: Multi-dimensional tensor class with efficient operations
- **Layer Architecture**: Modular layer system with forward/backward propagation
- **Model Building**: Sequential model container for easy network construction
- **Dataset Handling**: Abstract dataset interface with MNIST implementation
- **Training Infrastructure**: Complete training loop with callbacks and metrics
- **Optimizers**: SGD, Adam, RMSprop, and AdamW implementations
- **Loss Functions**: Cross-entropy, MSE, BCE, and label smoothing
- **Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix

### Neural Network Layers
- **Convolutional**: Conv2D with padding, stride, and bias support
- **Pooling**: MaxPool2D for spatial dimension reduction
- **Dense**: Fully connected layers with various initialization methods
- **Activations**: ReLU, Softmax, Sigmoid, Tanh
- **Future**: BatchNorm, Dropout, ResNet blocks (planned)

### Training Features
- **Callbacks**: Early stopping, learning rate scheduling, model checkpointing
- **Metrics**: Comprehensive evaluation metrics
- **Visualization**: Training progress tracking and logging
- **Checkpointing**: Model saving and loading capabilities

## 📁 Project Structure

```
cppvision/
├── include/cppvision/
│   ├── core/           # Core tensor and layer abstractions
│   ├── layers/         # Neural network layer implementations
│   ├── models/         # Model containers and architectures
│   ├── datasets/       # Dataset loading and preprocessing
│   ├── training/       # Training infrastructure
│   └── optimizers/     # Optimization algorithms
├── src/cppvision/      # Implementation files (with TODO comments)
├── examples/           # Usage examples
├── tests/              # Unit tests (planned)
└── docs/               # Documentation
```

## 🛠️ Building the Framework

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15 or higher
- MNIST dataset files (included in `data/` directory)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/cppvision.git
cd cppvision

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run the example
./mnist_cnn
```

### Build Options

```bash
# Enable tests
cmake -DBUILD_TESTS=ON ..

# Enable examples
cmake -DBUILD_EXAMPLES=ON ..

# Build with debug information
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## 🎯 Quick Start

### Basic CNN Example

```cpp
#include "cppvision/cppvision.h"

using namespace cppvision;

int main() {
    // Create CNN model
    auto model = std::make_unique<Sequential>("MNIST_CNN");
    
    model->add_layer(std::make_unique<Conv2D>(1, 32, 3, 1, 1));
    model->add_layer(std::make_unique<ReLU>());
    model->add_layer(std::make_unique<MaxPool2D>(2));
    
    model->add_layer(std::make_unique<Conv2D>(32, 64, 3, 1, 1));
    model->add_layer(std::make_unique<ReLU>());
    model->add_layer(std::make_unique<MaxPool2D>(2));
    
    model->add_layer(std::make_unique<Dense>(1600, 128));
    model->add_layer(std::make_unique<ReLU>());
    model->add_layer(std::make_unique<Dense>(128, 10));
    model->add_layer(std::make_unique<Softmax>());
    
    // Load dataset
    auto train_dataset = std::make_unique<MNISTDataset>("data/train-images-idx3-ubyte", 
                                                        "data/train-labels-idx1-ubyte");
    auto val_dataset = std::make_unique<MNISTDataset>("data/t10k-images-idx3-ubyte", 
                                                      "data/t10k-labels-idx1-ubyte");
    
    // Create trainer
    Trainer trainer(
        std::move(model),
        std::move(train_dataset),
        std::move(val_dataset),
        std::make_unique<Adam>(0.001),
        std::make_unique<CrossEntropyLoss>(),
        32, 10, true, true
    );
    
    // Add callbacks
    trainer.add_callback(std::make_unique<EarlyStopping>(5, 1e-4));
    trainer.add_callback(std::make_unique<ModelCheckpoint>("model_{epoch}.cpp"));
    
    // Train the model
    trainer.train();
    
    return 0;
}
```

## 📚 Implementation Status

### ✅ Completed (Skeleton)
- [x] Core tensor class with operations
- [x] Layer abstraction and inheritance
- [x] Dense, Conv2D, MaxPool2D, Activation layers
- [x] Sequential model container
- [x] Dataset abstraction and MNIST implementation
- [x] Training infrastructure with callbacks
- [x] Loss functions (CrossEntropy, MSE, BCE)
- [x] Metrics (Accuracy, Precision, Recall, F1)
- [x] Optimizers (SGD, Adam, RMSprop, AdamW)
- [x] Build system and examples

### 🔄 TODO (Implementation)
- [ ] Implement all algorithm details in .cpp files
- [ ] Add comprehensive unit tests
- [ ] Implement additional layers (BatchNorm, Dropout)
- [ ] Add more datasets (CIFAR-10, Fashion-MNIST)
- [ ] Implement advanced optimizers (AdaGrad, AdaDelta)
- [ ] Add visualization tools
- [ ] Performance optimization and profiling
- [ ] Documentation and tutorials

## 🎓 Learning Objectives

This framework is designed for educational purposes and provides:

1. **Deep Understanding**: Implement algorithms from scratch to understand the fundamentals
2. **Research Platform**: Extensible architecture for experimentation
3. **Performance Focus**: C++ implementation for efficiency
4. **Modern C++**: Uses C++17 features and best practices
5. **Modular Design**: Clean separation of concerns

## 🔬 Research Applications

The framework supports various research directions:

### Phase 1: Core Framework
- Modular deep learning framework
- PyTorch-like API design
- Comprehensive layer implementations

### Phase 2: Empirical Studies
- Generalization and robustness experiments
- Optimizer comparison studies
- Regularization effect analysis

### Phase 3: Systems Research
- Automatic differentiation engine
- Computational graph implementation
- Memory optimization techniques

### Phase 4: Scaling Studies
- Neural scaling laws
- Computational efficiency analysis
- Model compression techniques

### Phase 5: Domain Applications
- Astronomical data analysis
- Scientific computing applications
- Cross-domain transfer learning

## 🤝 Contributing

This is an educational project designed for learning and experimentation. Contributions are welcome:

1. **Algorithm Implementation**: Complete the TODO comments in .cpp files
2. **New Layers**: Add additional layer types
3. **Optimizers**: Implement more optimization algorithms
4. **Datasets**: Add support for more datasets
5. **Tests**: Add comprehensive unit tests
6. **Documentation**: Improve documentation and examples

## 📖 Documentation

- [API Reference](docs/api.md) - Complete API documentation
- [Tutorials](docs/tutorials.md) - Step-by-step tutorials
- [Examples](examples/) - Usage examples and demos
- [Research Roadmap](ROADMAP.md) - Long-term research directions

## 🏗️ Architecture

### Design Principles
- **Modularity**: Clean separation between components
- **Extensibility**: Easy to add new layers, optimizers, datasets
- **Performance**: Efficient C++ implementation
- **Usability**: PyTorch-inspired API design
- **Educational**: Clear code structure for learning

### Key Components

#### Tensor Class
```cpp
class Tensor {
    // Multi-dimensional data storage
    // Mathematical operations (+, -, *, matmul)
    // Shape manipulation (reshape, transpose)
    // Memory-efficient operations
};
```

#### Layer System
```cpp
class Layer {
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor> get_parameters() const = 0;
    // ... other methods
};
```

#### Training Infrastructure
```cpp
class Trainer {
    // Complete training loop
    // Callback system
    // Metrics tracking
    // Model checkpointing
};
```

## 🚀 Future Roadmap

### Short Term (1-2 months)
- Complete algorithm implementations
- Add comprehensive tests
- Performance optimization
- Documentation improvements

### Medium Term (3-6 months)
- Additional layer types
- More datasets and optimizers
- Visualization tools
- GPU acceleration (CUDA)

### Long Term (6+ months)
- Automatic differentiation engine
- Advanced architectures (ResNet, Transformer)
- Distributed training
- Model deployment tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- PyTorch team for API inspiration
- Deep learning community for algorithms and techniques
- C++ community for modern language features

## 📞 Contact

For questions, suggestions, or contributions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This framework is designed for educational purposes. While it provides a solid foundation for deep learning research, it's not intended for production use without significant additional development and testing.
