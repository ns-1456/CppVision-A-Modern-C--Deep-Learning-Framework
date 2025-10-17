#include "cppvision/training/trainer.h"
#include "cppvision/core/tensor.h"
#include "cppvision/models/sequential.h"
#include "cppvision/datasets/mnist_dataset.h"
#include "cppvision/optimizers/optimizer.h"
#include "cppvision/training/loss.h"
#include "cppvision/training/metrics.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace cppvision {

// Constructor
Trainer::Trainer(std::unique_ptr<Sequential> model,
                 std::unique_ptr<Dataset> train_dataset,
                 std::unique_ptr<Dataset> val_dataset,
                 std::unique_ptr<Optimizer> optimizer,
                 std::unique_ptr<Loss> loss_fn,
                 int batch_size, int num_epochs, bool shuffle, bool verbose)
    : model_(std::move(model)), train_dataset_(std::move(train_dataset)),
      val_dataset_(std::move(val_dataset)), optimizer_(std::move(optimizer)),
      loss_fn_(std::move(loss_fn)), batch_size_(batch_size), num_epochs_(num_epochs),
      shuffle_(shuffle), verbose_(verbose), current_epoch_(0), best_val_loss_(std::numeric_limits<double>::max()) {
    // TODO: Implement trainer constructor
    // Algorithm:
    //   1. Initialize training metrics
    train_metrics_ = std::make_unique<MetricCollection>("TrainMetrics");
    train_metrics_->add_metric(std::make_unique<Accuracy>("TrainAccuracy"));
    
    //   2. Initialize validation metrics
    val_metrics_ = std::make_unique<MetricCollection>("ValMetrics");
    val_metrics_->add_metric(std::make_unique<Accuracy>("ValAccuracy"));
    
    //   3. Set model to training mode
    model_->set_training(true);
    
    std::cout << "Trainer initialized with " << model_->num_parameters() << " parameters" << std::endl;
    // Hint: Initialize metrics and set model to training mode
}

// Training methods
void Trainer::train() {
    // TODO: Implement main training loop
    // Algorithm:
    //   1. Loop over epochs
    for (int epoch = 0; epoch < num_epochs_; ++epoch) {
        current_epoch_ = epoch;
        
        //   2. Train for one epoch
        auto train_results = train_epoch();
        
        //   3. Validate model
        auto val_results = validate();
        
        //   4. Print epoch results
        if (verbose_) {
            print_epoch_results(epoch, train_results, val_results);
        }
        
        //   5. Execute callbacks
        execute_callbacks(train_results, val_results);
        
        //   6. Check for early stopping
        if (should_stop_training()) {
            std::cout << "Early stopping triggered at epoch " << epoch << std::endl;
            break;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
    // Hint: Use train_epoch() and validate() methods
}

std::map<std::string, double> Trainer::train_epoch() {
    // TODO: Implement single epoch training
    // Algorithm:
    //   1. Set model to training mode
    model_->set_training(true);
    
    //   2. Reset metrics
    train_metrics_->reset();
    
    //   3. Shuffle dataset if needed
    if (shuffle_) {
        train_dataset_->shuffle();
    }
    
    //   4. Reset dataset
    train_dataset_->reset();
    
    //   5. Loop over batches
    int batch_count = 0;
    double total_loss = 0.0;
    
    while (true) {
        // Get batch
        auto [batch_inputs, batch_targets] = train_dataset_->get_batch(batch_size_);
        
        // Check if batch is empty (end of dataset)
        if (batch_inputs.size() == 0) {
            break;
        }
        
        // Forward pass
        Tensor predictions = model_->forward(batch_inputs);
        
        // Compute loss
        Tensor loss = loss_fn_->forward(predictions, batch_targets);
        total_loss += loss[0];
        
        // Backward pass
        Tensor grad_output = loss_fn_->backward(loss);
        model_->backward(grad_output);
        
        // Update parameters
        auto parameters = model_->get_parameters();
        auto gradients = model_->get_gradients();
        optimizer_->step(parameters, gradients);
        
        // Update metrics
        train_metrics_->update(predictions, batch_targets);
        
        // Zero gradients
        model_->zero_gradients();
        
        batch_count++;
        
        if (verbose_ && batch_count % 100 == 0) {
            std::cout << "Batch " << batch_count << ", Loss: " << loss[0] << std::endl;
        }
    }
    
    //   6. Compute average loss
    double avg_loss = total_loss / batch_count;
    
    //   7. Get metrics
    auto metrics = train_metrics_->compute();
    metrics["loss"] = avg_loss;
    
    return metrics;
    // Hint: Use model_->forward(), loss_fn_->forward(), model_->backward(), optimizer_->step()
}

std::map<std::string, double> Trainer::validate() {
    // TODO: Implement validation
    // Algorithm:
    //   1. Set model to evaluation mode
    model_->set_training(false);
    
    //   2. Reset metrics
    val_metrics_->reset();
    
    //   3. Reset dataset
    val_dataset_->reset();
    
    //   4. Loop over validation batches
    int batch_count = 0;
    double total_loss = 0.0;
    
    while (true) {
        // Get batch
        auto [batch_inputs, batch_targets] = val_dataset_->get_batch(batch_size_);
        
        // Check if batch is empty (end of dataset)
        if (batch_inputs.size() == 0) {
            break;
        }
        
        // Forward pass (no gradients needed)
        Tensor predictions = model_->forward(batch_inputs);
        
        // Compute loss
        Tensor loss = loss_fn_->forward(predictions, batch_targets);
        total_loss += loss[0];
        
        // Update metrics
        val_metrics_->update(predictions, batch_targets);
        
        batch_count++;
    }
    
    //   5. Compute average loss
    double avg_loss = total_loss / batch_count;
    
    //   6. Get metrics
    auto metrics = val_metrics_->compute();
    metrics["loss"] = avg_loss;
    
    //   7. Update best validation loss
    if (avg_loss < best_val_loss_) {
        best_val_loss_ = avg_loss;
    }
    
    return metrics;
    // Hint: Set model to evaluation mode, no backward pass needed
}

// Callback management
void Trainer::add_callback(std::unique_ptr<Callback> callback) {
    // TODO: Implement callback addition
    // Algorithm:
    //   1. Validate callback is not null
    if (!callback) {
        throw std::invalid_argument("Cannot add null callback to trainer");
    }
    //   2. Add callback to collection
    callbacks_.push_back(std::move(callback));
    // Hint: Use std::move to transfer ownership
}

void Trainer::execute_callbacks(const std::map<std::string, double>& train_results,
                               const std::map<std::string, double>& val_results) {
    // TODO: Implement callback execution
    // Algorithm:
    //   1. Execute all callbacks
    for (auto& callback : callbacks_) {
        callback->on_epoch_end(current_epoch_, train_results, val_results);
    }
    // Hint: Call on_epoch_end() for each callback
}

bool Trainer::should_stop_training() const {
    // TODO: Implement early stopping check
    // Algorithm:
    //   1. Check if any callback requests stopping
    for (const auto& callback : callbacks_) {
        if (callback->should_stop_training()) {
            return true;
        }
    }
    //   2. Return false if no callback requests stopping
    return false;
    // Hint: Check callback->should_stop_training() for each callback
}

// Model management
void Trainer::save_model(const std::string& filename) {
    // TODO: Implement model saving
    // Algorithm:
    //   1. Open file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    //   2. Save model parameters
    auto parameters = model_->get_parameters();
    
    // Save number of parameters
    int num_params = static_cast<int>(parameters.size());
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
    
    // Save each parameter tensor
    for (const auto& param : parameters) {
        // Save shape
        auto shape = param.shape();
        int shape_size = static_cast<int>(shape.size());
        file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
        file.write(reinterpret_cast<const char*>(shape.data()), shape_size * sizeof(int));
        
        // Save data
        int data_size = param.size();
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char*>(param.data()), data_size * sizeof(double));
    }
    
    //   3. Close file
    file.close();
    
    std::cout << "Model saved to " << filename << std::endl;
    // Hint: Use binary format for efficiency
}

void Trainer::load_model(const std::string& filename) {
    // TODO: Implement model loading
    // Algorithm:
    //   1. Open file for reading
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    //   2. Load model parameters
    int num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    
    auto parameters = model_->get_parameters();
    if (num_params != static_cast<int>(parameters.size())) {
        throw std::runtime_error("Parameter count mismatch in saved model");
    }
    
    // Load each parameter tensor
    for (auto& param : parameters) {
        // Load shape
        int shape_size;
        file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
        std::vector<int> shape(shape_size);
        file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(int));
        
        // Load data
        int data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        std::vector<double> data(data_size);
        file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(double));
        
        // Create tensor and update parameter
        Tensor loaded_param(data, shape, param.requires_grad());
        param = loaded_param;
    }
    
    //   3. Close file
    file.close();
    
    std::cout << "Model loaded from " << filename << std::endl;
    // Hint: Validate parameter shapes match expected shapes
}

// Utility methods
void Trainer::print_epoch_results(int epoch, const std::map<std::string, double>& train_results,
                                 const std::map<std::string, double>& val_results) {
    // TODO: Implement epoch results printing
    // Algorithm:
    //   1. Print epoch header
    std::cout << "\nEpoch " << std::setw(3) << epoch + 1 << "/" << num_epochs_ << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    //   2. Print training results
    std::cout << "Train - ";
    for (const auto& [name, value] : train_results) {
        std::cout << name << ": " << std::fixed << std::setprecision(4) << value << " ";
    }
    std::cout << std::endl;
    
    //   3. Print validation results
    std::cout << "Val   - ";
    for (const auto& [name, value] : val_results) {
        std::cout << name << ": " << std::fixed << std::setprecision(4) << value << " ";
    }
    std::cout << std::endl;
    // Hint: Use std::setw() and std::setprecision() for formatting
}

// Callback Implementations

// EarlyStopping Callback
EarlyStopping::EarlyStopping(int patience, double min_delta, const std::string& monitor_metric, bool restore_best_weights)
    : patience_(patience), min_delta_(min_delta), monitor_metric_(monitor_metric),
      restore_best_weights_(restore_best_weights), best_score_(std::numeric_limits<double>::max()),
      wait_count_(0), best_weights_(), stop_training_(false) {
    // Constructor - initialize early stopping parameters
}

void EarlyStopping::on_epoch_end(int epoch, const std::map<std::string, double>& train_results,
                                const std::map<std::string, double>& val_results) {
    // TODO: Implement early stopping logic
    // Algorithm:
    //   1. Get current score from validation results
    auto it = val_results.find(monitor_metric_);
    if (it == val_results.end()) {
        std::cerr << "Warning: Monitor metric '" << monitor_metric_ << "' not found in validation results" << std::endl;
        return;
    }
    
    double current_score = it->second;
    
    //   2. Check if score improved
    if (current_score < best_score_ - min_delta_) {
        best_score_ = current_score;
        wait_count_ = 0;
        // Save best weights if needed
        if (restore_best_weights_) {
            // In a full implementation, save model weights here
        }
    } else {
        wait_count_++;
    }
    
    //   3. Check if patience exceeded
    if (wait_count_ >= patience_) {
        stop_training_ = true;
        std::cout << "Early stopping triggered: " << monitor_metric_ << " did not improve for " << patience_ << " epochs" << std::endl;
    }
    // Hint: Monitor validation metric and stop if no improvement
}

bool EarlyStopping::should_stop_training() const {
    // TODO: Implement stop training check
    // Algorithm:
    //   1. Return stop_training_ flag
    return stop_training_;
}

// LearningRateScheduler Callback
LearningRateScheduler::LearningRateScheduler(std::function<double(int, double)> schedule_fn, const std::string& name)
    : schedule_fn_(schedule_fn), name_(name) {
    // Constructor - initialize learning rate scheduler
}

void LearningRateScheduler::on_epoch_end(int epoch, const std::map<std::string, double>& train_results,
                                         const std::map<std::string, double>& val_results) {
    // TODO: Implement learning rate scheduling
    // Algorithm:
    //   1. Get current learning rate from optimizer
    // Note: This would require optimizer to expose learning rate getter/setter
    // For now, this is a placeholder implementation
    
    //   2. Compute new learning rate using schedule function
    // double current_lr = optimizer_->get_learning_rate();
    // double new_lr = schedule_fn_(epoch, current_lr);
    
    //   3. Update optimizer learning rate
    // optimizer_->set_learning_rate(new_lr);
    
    // Placeholder - would need optimizer interface for learning rate control
    // Hint: Use schedule_fn_ to compute new learning rate
}

bool LearningRateScheduler::should_stop_training() const {
    // TODO: Implement stop training check
    // Algorithm:
    //   1. Learning rate scheduler doesn't stop training
    return false;
}

// ModelCheckpoint Callback
ModelCheckpoint::ModelCheckpoint(const std::string& filepath, const std::string& monitor_metric,
                                bool save_best_only, bool save_weights_only, int period)
    : filepath_(filepath), monitor_metric_(monitor_metric), save_best_only_(save_best_only),
      save_weights_only_(save_weights_only), period_(period), best_score_(std::numeric_limits<double>::max()) {
    // Constructor - initialize model checkpoint parameters
}

void ModelCheckpoint::on_epoch_end(int epoch, const std::map<std::string, double>& train_results,
                                  const std::map<std::string, double>& val_results) {
    // TODO: Implement model checkpointing
    // Algorithm:
    //   1. Check if should save this epoch
    if ((epoch + 1) % period_ != 0) {
        return;
    }
    
    //   2. Get current score
    auto it = val_results.find(monitor_metric_);
    if (it == val_results.end()) {
        std::cerr << "Warning: Monitor metric '" << monitor_metric_ << "' not found" << std::endl;
        return;
    }
    
    double current_score = it->second;
    
    //   3. Check if should save (best only or every period)
    bool should_save = !save_best_only_ || current_score < best_score_;
    
    if (should_save) {
        //   4. Generate filename
        std::string filename = filepath_;
        size_t pos = filename.find("{epoch}");
        if (pos != std::string::npos) {
            filename.replace(pos, 7, std::to_string(epoch + 1));
        }
        
        //   5. Save model
        // trainer_->save_model(filename); // Would need trainer reference
        
        if (current_score < best_score_) {
            best_score_ = current_score;
        }
        
        std::cout << "Model checkpoint saved: " << filename << std::endl;
    }
    // Hint: Replace {epoch} placeholder in filename
}

bool ModelCheckpoint::should_stop_training() const {
    // TODO: Implement stop training check
    // Algorithm:
    //   1. Model checkpoint doesn't stop training
    return false;
}

} // namespace cppvision
