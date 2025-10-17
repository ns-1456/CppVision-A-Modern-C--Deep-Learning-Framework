#include "cppvision/models/sequential.h"
#include "cppvision/core/layer.h"
#include "cppvision/layers/dense.h"
#include <iostream>
#include <sstream>

namespace cppvision {

// Constructor
Sequential::Sequential(const std::string& name) : name_(name) {
    // Constructor - initialize empty layer container
}

// Layer management
void Sequential::add_layer(std::unique_ptr<Layer> layer) {
    // TODO: Implement layer addition
    // Algorithm:
    //   1. Validate layer is not null
    if (!layer) {
        throw std::invalid_argument("Cannot add null layer to Sequential model");
    }
    //   2. Add layer to layers_ container
    layers_.push_back(std::move(layer));
    //   3. Update layer training mode if needed
    // Hint: Use std::move to transfer ownership
}

Layer& Sequential::get_layer(int index) {
    // TODO: Implement layer access
    // Algorithm:
    //   1. Validate index is within bounds
    if (index < 0 || index >= static_cast<int>(layers_.size())) {
        throw std::out_of_range("Layer index " + std::to_string(index) + " is out of bounds [0, " + std::to_string(layers_.size()) + ")");
    }
    //   2. Return reference to layer at index
    return *layers_[index];
    // Hint: Check index >= 0 && index < layers_.size()
}

const Layer& Sequential::get_layer(int index) const {
    // TODO: Implement const layer access
    // Algorithm:
    //   1. Validate index is within bounds
    if (index < 0 || index >= static_cast<int>(layers_.size())) {
        throw std::out_of_range("Layer index " + std::to_string(index) + " is out of bounds [0, " + std::to_string(layers_.size()) + ")");
    }
    //   2. Return const reference to layer at index
    return *layers_[index];
}

// Forward and backward propagation
Tensor Sequential::forward(const Tensor& input) {
    // TODO: Implement forward pass through all layers
    // Algorithm:
    //   1. Start with input tensor
    Tensor current_input = input;
    //   2. Pass through each layer in sequence
    for (auto& layer : layers_) {
        current_input = layer->forward(current_input);
    }
    //   3. Return output from last layer
    return current_input;
    // Hint: Use layer->forward() for each layer
}

Tensor Sequential::backward(const Tensor& grad_output) {
    // TODO: Implement backward pass through all layers
    // Algorithm:
    //   1. Start with grad_output
    Tensor current_grad = grad_output;
    //   2. Pass through each layer in reverse order
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
    //   3. Return gradient to first layer
    return current_grad;
    // Hint: Use layer->backward() for each layer in reverse
}

// Parameter management
std::vector<Tensor> Sequential::get_parameters() const {
    // TODO: Implement parameter collection
    // Algorithm:
    //   1. Create vector to store all parameters
    std::vector<Tensor> all_parameters;
    //   2. Iterate through all layers
    for (const auto& layer : layers_) {
        //   3. Collect parameters from each layer
        std::vector<Tensor> layer_params = layer->get_parameters();
        all_parameters.insert(all_parameters.end(), layer_params.begin(), layer_params.end());
    }
    //   4. Return combined parameter vector
    return all_parameters;
}

std::vector<Tensor> Sequential::get_gradients() const {
    // TODO: Implement gradient collection
    // Algorithm:
    //   1. Create vector to store all gradients
    std::vector<Tensor> all_gradients;
    //   2. Iterate through all layers
    for (const auto& layer : layers_) {
        //   3. Collect gradients from each layer
        std::vector<Tensor> layer_grads = layer->get_gradients();
        all_gradients.insert(all_gradients.end(), layer_grads.begin(), layer_grads.end());
    }
    //   4. Return combined gradient vector
    return all_gradients;
}

void Sequential::update_parameters(const std::vector<Tensor>& gradients) {
    // TODO: Implement parameter update
    // Algorithm:
    //   1. Track gradient index
    size_t grad_index = 0;
    //   2. Iterate through all layers
    for (auto& layer : layers_) {
        //   3. Update parameters for each layer
        std::vector<Tensor> layer_gradients;
        std::vector<Tensor> layer_params = layer->get_parameters();
        
        // Extract gradients for this layer
        for (size_t i = 0; i < layer_params.size(); ++i) {
            if (grad_index < gradients.size()) {
                layer_gradients.push_back(gradients[grad_index]);
                grad_index++;
            }
        }
        
        // Update layer parameters
        if (!layer_gradients.empty()) {
            layer->update_parameters(layer_gradients);
        }
    }
    //   4. Advance gradient index appropriately
    // Hint: Each layer knows how many parameters it has
}

void Sequential::zero_gradients() {
    // TODO: Implement gradient zeroing
    // Algorithm:
    //   1. Iterate through all layers
    for (auto& layer : layers_) {
        //   2. Zero gradients for each layer
        layer->zero_gradients();
    }
    // Hint: Use layer->zero_gradients()
}

// Layer information
int Sequential::num_parameters() const {
    // TODO: Implement parameter counting
    // Algorithm:
    //   1. Sum parameters from all layers
    int total_params = 0;
    for (const auto& layer : layers_) {
        total_params += layer->num_parameters();
    }
    //   2. Return total parameter count
    return total_params;
}

bool Sequential::has_parameters() const {
    // TODO: Implement parameter check
    // Algorithm:
    //   1. Check if any layer has parameters
    for (const auto& layer : layers_) {
        if (layer->num_parameters() > 0) {
            return true;
        }
    }
    //   2. Return true if any layer has parameters
    return false;
}

// Training mode
void Sequential::set_training(bool training) {
    // TODO: Implement training mode setting
    // Algorithm:
    //   1. Set training_mode_ for this layer
    training_mode_ = training;
    //   2. Set training mode for all child layers
    for (auto& layer : layers_) {
        layer->set_training(training);
    }
    // Hint: Use layer->set_training(training) for each layer
}

// Model information and utilities
void Sequential::summary() const {
    // TODO: Implement model summary
    // Algorithm:
    //   1. Print model name and total parameters
    //   2. Print each layer with its parameters
    //   3. Show input/output shapes if available
    std::cout << "Model: " << name_ << std::endl;
    std::cout << "Total parameters: " << num_parameters() << std::endl;
    std::cout << "Number of layers: " << layers_.size() << std::endl;
    std::cout << "===========================================" << std::endl;
    
    for (size_t i = 0; i < layers_.size(); ++i) {
        std::cout << "Layer " << i << ": " << layers_[i]->get_name() << std::endl;
        std::cout << "  Parameters: " << layers_[i]->num_parameters() << std::endl;
        // layers_[i]->summary(); // Commented out - summary() not in base class
        std::cout << std::endl;
    }
    // TODO: Add layer-by-layer summary
}

std::string Sequential::get_summary() const {
    // TODO: Implement string summary
    // Algorithm:
    //   1. Create string with model information
    std::ostringstream oss;
    oss << "Sequential Model: " << name_ << "\n";
    oss << "Total parameters: " << num_parameters() << "\n";
    oss << "Number of layers: " << layers_.size() << "\n";
    //   2. Include layer details
    for (size_t i = 0; i < layers_.size(); ++i) {
        oss << "Layer " << i << ": " << layers_[i]->get_name() 
            << " (" << layers_[i]->num_parameters() << " parameters)\n";
    }
    //   3. Return formatted string
    return oss.str();
}

std::vector<std::vector<int>> Sequential::get_input_shapes() const {
    // TODO: Implement input shape calculation
    // Algorithm:
    //   1. Calculate input shapes for each layer
    std::vector<std::vector<int>> input_shapes;
    //   2. Return vector of input shapes
    // Hint: This requires knowing the input shape to the model
    return input_shapes; // Placeholder - would need input shape to calculate
}

std::vector<int> Sequential::get_output_shape(const std::vector<int>& input_shape) const {
    // TODO: Implement output shape calculation
    // Algorithm:
    //   1. Start with input shape
    std::vector<int> current_shape = input_shape;
    //   2. Pass through each layer to calculate output shape
    for (const auto& layer : layers_) {
        // For now, return input shape as placeholder
        // In a full implementation, each layer would have a method to calculate output shape
        current_shape = current_shape; // Placeholder
    }
    //   3. Return final output shape
    return current_shape;
    // Hint: Each layer should have a method to calculate output shape
}

// Model saving and loading
void Sequential::save_parameters(const std::string& filename) const {
    // TODO: Implement parameter saving
    // Algorithm:
    //   1. Open file for writing
    //   2. Save model metadata (name, num_layers, etc.)
    //   3. Save parameters for each layer
    //   4. Close file
    // Hint: Use binary format for efficiency
    // Placeholder implementation
}

void Sequential::load_parameters(const std::string& filename) {
    // TODO: Implement parameter loading
    // Algorithm:
    //   1. Open file for reading
    //   2. Load model metadata
    //   3. Load parameters for each layer
    //   4. Close file
    // Hint: Validate file format and parameter shapes
    // Placeholder implementation
}

// Private helper methods
bool Sequential::validate_model() const {
    // Model validation implementation
    // Algorithm:
    //   1. Check that all layers are valid
    for (const auto& layer : layers_) {
        if (!layer) {
            return false;
        }
    }
    //   2. Check that layer connections are compatible
    //   3. Return true if model is valid
    return true; // Placeholder - would need shape validation
}

int Sequential::count_layer_parameters(const Layer& layer) const {
    // Layer parameter counting implementation
    // Algorithm:
    //   1. Return number of parameters in layer
    return layer.num_parameters();
}

} // namespace cppvision
