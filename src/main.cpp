#include <iostream>
#include <memory>
#include <string>

// CppVision Framework includes
#include "cppvision/core/tensor.h"
#include "cppvision/core/layer.h"
#include "cppvision/layers/dense.h"
#include "cppvision/layers/conv2d.h"
#include "cppvision/layers/maxpool2d.h"
#include "cppvision/layers/activations.h"
#include "cppvision/models/sequential.h"
#include "cppvision/training/loss.h"
#include "cppvision/training/metrics.h"
#include "cppvision/optimizers/sgd.h"
#include "cppvision/optimizers/adam.h"

using namespace cppvision;

int main() {
    std::cout << "=== CppVision Framework Demo ===" << std::endl;
    std::cout << "Building CNN for MNIST Classification" << std::endl;
    
    try {
        // 1. Create CNN Model Architecture
        std::cout << "\n[1] Creating CNN Model..." << std::endl;
        
        auto model = std::make_unique<Sequential>("MNIST_CNN");
        
        // Add layers to the model
        model->add_layer(std::make_unique<Conv2D>(1, 32, 3, 1, 1, true, "Conv1"));
        model->add_layer(std::make_unique<ReLU>("ReLU1"));
        model->add_layer(std::make_unique<MaxPool2D>(2, 2, 0, "MaxPool1"));
        
        model->add_layer(std::make_unique<Conv2D>(32, 64, 3, 1, 1, true, "Conv2"));
        model->add_layer(std::make_unique<ReLU>("ReLU2"));
        model->add_layer(std::make_unique<MaxPool2D>(2, 2, 0, "MaxPool2"));
        
        model->add_layer(std::make_unique<Dense>(3136, 128, true, "Dense1"));
        model->add_layer(std::make_unique<ReLU>("ReLU3"));
        model->add_layer(std::make_unique<Dense>(128, 10, true, "Dense2"));
        model->add_layer(std::make_unique<Softmax>(-1, "Softmax"));
        
        std::cout << "Model created with " << model->num_parameters() << " parameters" << std::endl;
        model->summary();

        // 2. Create Loss Function and Optimizer
        std::cout << "\n[2] Setting up Training Components..." << std::endl;
        
        auto loss_fn = std::make_unique<CrossEntropyLoss>(1e-15, "CrossEntropy");
        auto optimizer = std::make_unique<Adam>(0.001, 0.9, 0.999, 1e-8, 0.0, "Adam");
        
        std::cout << "Loss function: " << loss_fn->get_name() << std::endl;
        std::cout << "Optimizer: " << optimizer->get_name() << std::endl;
        std::cout << "Learning rate: " << optimizer->get_learning_rate() << std::endl;

        // 3. Create Metrics
        std::cout << "\n[3] Setting up Metrics..." << std::endl;
        
        auto train_metrics = std::make_unique<MetricCollection>();
        train_metrics->add_metric(std::make_unique<Accuracy>("TrainAccuracy"));
        
        auto val_metrics = std::make_unique<MetricCollection>();
        val_metrics->add_metric(std::make_unique<Accuracy>("ValAccuracy"));
        val_metrics->add_metric(std::make_unique<Precision>(10, "ValPrecision"));
        val_metrics->add_metric(std::make_unique<Recall>(10, "ValRecall"));
        val_metrics->add_metric(std::make_unique<F1Score>(10, "ValF1"));

        // 4. Test Basic Functionality
        std::cout << "\n[4] Testing Basic Functionality..." << std::endl;
        
        // Create a small test batch
        std::vector<int> test_shape = {2, 10};
        Tensor test_predictions = Tensor::random_normal(test_shape, 0.0, 1.0, false);
        Tensor test_targets = Tensor::zeros(test_shape, false);
        
        // Set some test targets
        test_targets[0] = 1.0;  // First sample: class 0
        test_targets[11] = 1.0; // Second sample: class 1
        
        // Test loss computation
        double loss = loss_fn->compute_loss(test_predictions, test_targets);
        std::cout << "Test loss: " << loss << std::endl;
        
        // Test metrics
        train_metrics->update(test_predictions, test_targets);
        auto metrics = train_metrics->compute();
        std::cout << "Test metrics:" << std::endl;
        for (const auto& [name, value] : metrics) {
            std::cout << "  " << name << ": " << value << std::endl;
        }

        // 5. Test Model Forward Pass
        std::cout << "\n[5] Testing Model Forward Pass..." << std::endl;
        
        // Create a test input
        std::vector<int> input_shape = {1, 1, 28, 28};
        Tensor test_input = Tensor::random_normal(input_shape, 0.0, 1.0, false);
        
        try {
            Tensor output = model->forward(test_input);
            std::cout << "Model forward pass successful!" << std::endl;
            std::cout << "Output shape: ";
            for (int dim : output.shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Model forward pass failed: " << e.what() << std::endl;
        }

        std::cout << "\n=== Demo Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

/*
Example Usage Instructions:

1. Build the project:
   mkdir build && cd build
   cmake ..
   make

2. Run the demo:
   ./mnist_cnn

3. Expected output:
   - Model architecture summary
   - Basic functionality tests
   - Model forward pass test

4. Framework Features Demonstrated:
   - Sequential model creation
   - CNN architecture (Conv2D + ReLU + MaxPool + Dense + Softmax)
   - Loss function computation
   - Metrics computation
   - Model forward pass

5. Next Steps for Implementation:
   - Fix remaining compilation issues
   - Implement Trainer class
   - Add training loop
   - Add MNIST dataset loading
   - Add model saving/loading
*/