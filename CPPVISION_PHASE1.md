# CppVision Framework: Phase 1 Implementation Plan

## Current Codebase Analysis

Your existing implementation has:
- ✅ **Data Pipeline**: MNIST loading and preprocessing
- ✅ **Math Utils**: Core linear algebra operations
- ✅ **Dense Layer**: Basic structure with forward/backward design
- ✅ **Build System**: CMake ready for expansion

## Phase 1: CppVision Framework Architecture

### 1. Core Layer Abstraction

```cpp
// include/cppvision/core/layer.h
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor> get_parameters() const = 0;
    virtual std::vector<Tensor> get_gradients() const = 0;
    virtual void update_parameters(const std::vector<Tensor>& gradients) = 0;
    virtual std::string get_name() const = 0;
};
```

### 2. Tensor Implementation

```cpp
// include/cppvision/core/tensor.h
class Tensor {
private:
    std::vector<double> data_;
    std::vector<int> shape_;
    bool requires_grad_;
    
public:
    Tensor(const std::vector<int>& shape, bool requires_grad = false);
    Tensor(const std::vector<double>& data, const std::vector<int>& shape);
    
    // Accessors
    double& operator()(const std::vector<int>& indices);
    const double& operator()(const std::vector<int>& indices) const;
    
    // Shape operations
    Tensor reshape(const std::vector<int>& new_shape);
    Tensor transpose();
    
    // Mathematical operations
    Tensor operator+(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor matmul(const Tensor& other);
};
```

### 3. Layer Implementations

#### Dense Layer (Enhanced from your current implementation)
```cpp
// include/cppvision/layers/dense.h
class Dense : public Layer {
private:
    Tensor weights_;
    Tensor biases_;
    Tensor weight_gradients_;
    Tensor bias_gradients_;
    Tensor input_cache_;
    
public:
    Dense(int input_size, int output_size, bool use_bias = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void update_parameters(const std::vector<Tensor>& gradients) override;
    std::string get_name() const override { return "Dense"; }
};
```

#### Conv2D Layer
```cpp
// include/cppvision/layers/conv2d.h
class Conv2D : public Layer {
private:
    Tensor kernels_;  // [out_channels, in_channels, kernel_h, kernel_w]
    Tensor biases_;
    int stride_;
    int padding_;
    
public:
    Conv2D(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    // ... other required methods
};
```

#### Activation Layers
```cpp
// include/cppvision/layers/activations.h
class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string get_name() const override { return "ReLU"; }
};

class Softmax : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string get_name() const override { return "Softmax"; }
};
```

### 4. Model Architecture

```cpp
// include/cppvision/models/sequential.h
class Sequential : public Layer {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    
public:
    Sequential() = default;
    
    void add_layer(std::unique_ptr<Layer> layer);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    // Training utilities
    std::vector<Tensor> get_all_parameters();
    std::vector<Tensor> get_all_gradients();
    void zero_gradients();
    void update_all_parameters(const std::vector<Tensor>& gradients);
};
```

### 5. Dataset Abstraction

```cpp
// include/cppvision/datasets/dataset.h
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual std::pair<Tensor, Tensor> get_batch(int batch_size) = 0;
    virtual int size() const = 0;
    virtual void shuffle() = 0;
};

// include/cppvision/datasets/mnist_dataset.h
class MNISTDataset : public Dataset {
private:
    std::vector<Tensor> images_;
    std::vector<Tensor> labels_;
    int current_index_;
    
public:
    MNISTDataset(const std::string& images_path, const std::string& labels_path);
    std::pair<Tensor, Tensor> get_batch(int batch_size) override;
    int size() const override;
    void shuffle() override;
};
```

### 6. Training Infrastructure

```cpp
// include/cppvision/training/trainer.h
class Trainer {
private:
    std::unique_ptr<Sequential> model_;
    std::unique_ptr<Dataset> train_dataset_;
    std::unique_ptr<Dataset> val_dataset_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<Loss> loss_fn_;
    
public:
    Trainer(std::unique_ptr<Sequential> model,
            std::unique_ptr<Dataset> train_dataset,
            std::unique_ptr<Dataset> val_dataset,
            std::unique_ptr<Optimizer> optimizer,
            std::unique_ptr<Loss> loss_fn);
    
    void train(int epochs);
    double validate();
    void save_model(const std::string& path);
    void load_model(const std::string& path);
};
```

### 7. Optimizers

```cpp
// include/cppvision/optimizers/optimizer.h
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<Tensor>& parameters,
                     const std::vector<Tensor>& gradients) = 0;
    virtual void zero_grad() = 0;
};

// include/cppvision/optimizers/sgd.h
class SGD : public Optimizer {
private:
    double learning_rate_;
    double momentum_;
    std::vector<Tensor> velocity_;
    
public:
    SGD(double learning_rate = 0.01, double momentum = 0.0);
    void step(const std::vector<Tensor>& parameters,
              const std::vector<Tensor>& gradients) override;
    void zero_grad() override;
};
```

## Implementation Steps

### Step 1: Create Core Infrastructure (Week 1)
1. Implement `Tensor` class with basic operations
2. Create `Layer` base class
3. Enhance your existing `Dense` layer to inherit from `Layer`
4. Add `ReLU` and `Softmax` activation layers

### Step 2: Add Convolutional Layers (Week 1-2)
1. Implement `Conv2D` layer
2. Add `MaxPool2D` layer
3. Create `Sequential` model container
4. Test CNN architecture on MNIST

### Step 3: Training Infrastructure (Week 2)
1. Implement `Trainer` class
2. Add `SGD` optimizer
3. Create `CrossEntropyLoss`
4. Add metrics (accuracy, loss tracking)

### Step 4: Dataset Expansion (Week 2-3)
1. Abstract your MNIST loading into `MNISTDataset`
2. Add support for Fashion-MNIST
3. Implement data augmentation
4. Add batch loading and shuffling

### Step 5: Advanced Features (Week 3)
1. Add `Adam` optimizer
2. Implement callbacks (early stopping, learning rate scheduling)
3. Add model checkpointing
4. Create visualization utilities

## File Structure

```
include/cppvision/
├── core/
│   ├── tensor.h
│   ├── layer.h
│   └── utils.h
├── layers/
│   ├── dense.h
│   ├── conv2d.h
│   ├── maxpool2d.h
│   └── activations.h
├── models/
│   └── sequential.h
├── datasets/
│   ├── dataset.h
│   ├── mnist_dataset.h
│   └── fashion_mnist_dataset.h
├── training/
│   ├── trainer.h
│   ├── loss.h
│   └── metrics.h
├── optimizers/
│   ├── optimizer.h
│   ├── sgd.h
│   └── adam.h
└── utils/
    ├── visualization.h
    └── checkpoint.h
```

## Usage Example

```cpp
// Example usage of CppVision framework
int main() {
    // Create model
    auto model = std::make_unique<Sequential>();
    model->add_layer(std::make_unique<Conv2D>(1, 32, 3));
    model->add_layer(std::make_unique<ReLU>());
    model->add_layer(std::make_unique<MaxPool2D>(2));
    model->add_layer(std::make_unique<Conv2D>(32, 64, 3));
    model->add_layer(std::make_unique<ReLU>());
    model->add_layer(std::make_unique<MaxPool2D>(2));
    model->add_layer(std::make_unique<Dense>(1600, 128));
    model->add_layer(std::make_unique<ReLU>());
    model->add_layer(std::make_unique<Dense>(128, 10));
    model->add_layer(std::make_unique<Softmax>());
    
    // Create datasets
    auto train_dataset = std::make_unique<MNISTDataset>("train_images", "train_labels");
    auto val_dataset = std::make_unique<MNISTDataset>("test_images", "test_labels");
    
    // Create optimizer and loss
    auto optimizer = std::make_unique<SGD>(0.01);
    auto loss_fn = std::make_unique<CrossEntropyLoss>();
    
    // Create trainer
    Trainer trainer(std::move(model), std::move(train_dataset), 
                   std::move(val_dataset), std::move(optimizer), std::move(loss_fn));
    
    // Train model
    trainer.train(10);
    
    return 0;
}
```

This framework provides a solid foundation for all your subsequent research directions while maintaining the modularity and extensibility needed for advanced experiments.
