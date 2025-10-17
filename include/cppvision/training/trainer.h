#ifndef CPPVISION_TRAINING_TRAINER_H
#define CPPVISION_TRAINING_TRAINER_H

#include "../core/tensor.h"
#include "../models/sequential.h"
#include "../datasets/dataset.h"
#include "loss.h"
#include "metrics.h"
#include "../optimizers/optimizer.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <map>

namespace cppvision {

/**
 * @brief Training callback interface
 * 
 * Defines the interface for training callbacks that can be called
 * during different phases of training.
 */
class Callback {
public:
    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~Callback() = default;

    /**
     * @brief Called at the beginning of training
     * @param trainer Reference to trainer
     */
    virtual void on_train_begin(Trainer& trainer) {}

    /**
     * @brief Called at the end of training
     * @param trainer Reference to trainer
     */
    virtual void on_train_end(Trainer& trainer) {}

    /**
     * @brief Called at the beginning of an epoch
     * @param trainer Reference to trainer
     * @param epoch Current epoch number
     */
    virtual void on_epoch_begin(Trainer& trainer, int epoch) {}

    /**
     * @brief Called at the end of an epoch
     * @param trainer Reference to trainer
     * @param epoch Current epoch number
     * @param train_loss Training loss
     * @param val_loss Validation loss
     * @param train_metrics Training metrics
     * @param val_metrics Validation metrics
     */
    virtual void on_epoch_end(Trainer& trainer, int epoch, double train_loss, double val_loss,
                             const std::map<std::string, double>& train_metrics,
                             const std::map<std::string, double>& val_metrics) {}

    /**
     * @brief Called at the beginning of a batch
     * @param trainer Reference to trainer
     * @param batch_idx Current batch index
     * @param batch_size Current batch size
     */
    virtual void on_batch_begin(Trainer& trainer, int batch_idx, int batch_size) {}

    /**
     * @brief Called at the end of a batch
     * @param trainer Reference to trainer
     * @param batch_idx Current batch index
     * @param batch_size Current batch size
     * @param loss Current batch loss
     */
    virtual void on_batch_end(Trainer& trainer, int batch_idx, int batch_size, double loss) {}
};

/**
 * @brief Early stopping callback
 * 
 * Monitors validation loss and stops training if no improvement
 * is seen for a specified number of epochs.
 */
class EarlyStopping : public Callback {
private:
    int patience_;                 // Number of epochs to wait
    double min_delta_;            // Minimum change to qualify as improvement
    std::string monitor_;          // Metric to monitor
    int best_epoch_;              // Best epoch so far
    double best_score_;           // Best score so far
    int wait_count_;              // Number of epochs waited
    bool restore_best_weights_;   // Whether to restore best weights
    bool stopped_early_;          // Whether training was stopped early

public:
    /**
     * @brief Constructor for early stopping callback
     * @param patience Number of epochs to wait
     * @param min_delta Minimum change to qualify as improvement
     * @param monitor Metric to monitor
     * @param restore_best_weights Whether to restore best weights
     */
    explicit EarlyStopping(int patience = 10, double min_delta = 0.0,
                          const std::string& monitor = "val_loss",
                          bool restore_best_weights = true);

    /**
     * @brief Destructor
     */
    ~EarlyStopping() override = default;

    /**
     * @brief Called at the end of an epoch
     */
    void on_epoch_end(Trainer& trainer, int epoch, double train_loss, double val_loss,
                     const std::map<std::string, double>& train_metrics,
                     const std::map<std::string, double>& val_metrics) override;

    /**
     * @brief Check if training should stop
     * @return True if training should stop
     */
    bool should_stop() const { return stopped_early_; }
    
    /**
     * @brief Get best epoch
     * @return Best epoch number
     */
    int get_best_epoch() const { return best_epoch_; }
    
    /**
     * @brief Get best score
     * @return Best score
     */
    double get_best_score() const { return best_score_; }
};

/**
 * @brief Learning rate scheduler callback
 * 
 * Adjusts learning rate during training based on various schedules.
 */
class LearningRateScheduler : public Callback {
private:
    std::function<double(int, double)> schedule_func_;  // Schedule function
    std::string name_;                                   // Scheduler name

public:
    /**
     * @brief Constructor for learning rate scheduler
     * @param schedule_func Function that takes (epoch, current_lr) and returns new_lr
     * @param name Scheduler name
     */
    explicit LearningRateScheduler(std::function<double(int, double)> schedule_func,
                                  const std::string& name = "LRScheduler");

    /**
     * @brief Destructor
     */
    ~LearningRateScheduler() override = default;

    /**
     * @brief Called at the beginning of an epoch
     */
    void on_epoch_begin(Trainer& trainer, int epoch) override;

    // Static factory methods for common schedules
    /**
     * @brief Create step decay scheduler
     * @param step_size Number of epochs between decay
     * @param decay_factor Factor to multiply learning rate by
     * @return Learning rate scheduler
     */
    static std::unique_ptr<LearningRateScheduler> step_decay(int step_size, double decay_factor);
    
    /**
     * @brief Create exponential decay scheduler
     * @param decay_rate Decay rate
     * @return Learning rate scheduler
     */
    static std::unique_ptr<LearningRateScheduler> exponential_decay(double decay_rate);
    
    /**
     * @brief Create cosine annealing scheduler
     * @param T_max Maximum number of epochs
     * @param eta_min Minimum learning rate
     * @return Learning rate scheduler
     */
    static std::unique_ptr<LearningRateScheduler> cosine_annealing(int T_max, double eta_min = 0.0);
};

/**
 * @brief Model checkpoint callback
 * 
 * Saves model checkpoints during training at specified intervals.
 */
class ModelCheckpoint : public Callback {
private:
    std::string filepath_;         // File path template
    std::string monitor_;          // Metric to monitor
    bool save_best_only_;          // Whether to save only best model
    bool save_weights_only_;       // Whether to save only weights
    int period_;                   // Save frequency in epochs
    int last_save_epoch_;          // Last epoch when model was saved
    double best_score_;            // Best score so far

public:
    /**
     * @brief Constructor for model checkpoint callback
     * @param filepath File path template (can include {epoch} and {monitor} placeholders)
     * @param monitor Metric to monitor
     * @param save_best_only Whether to save only best model
     * @param save_weights_only Whether to save only weights
     * @param period Save frequency in epochs
     */
    explicit ModelCheckpoint(const std::string& filepath = "model_epoch_{epoch}.cpp",
                           const std::string& monitor = "val_loss",
                           bool save_best_only = false,
                           bool save_weights_only = true,
                           int period = 1);

    /**
     * @brief Destructor
     */
    ~ModelCheckpoint() override = default;

    /**
     * @brief Called at the end of an epoch
     */
    void on_epoch_end(Trainer& trainer, int epoch, double train_loss, double val_loss,
                     const std::map<std::string, double>& train_metrics,
                     const std::map<std::string, double>& val_metrics) override;

private:
    /**
     * @brief Generate filepath with placeholders replaced
     * @param epoch Current epoch
     * @param score Current score
     * @return Generated filepath
     */
    std::string generate_filepath(int epoch, double score) const;
};

/**
 * @brief Main trainer class for neural network training
 * 
 * Handles the complete training loop including forward pass, backward pass,
 * parameter updates, validation, and callback execution.
 */
class Trainer {
private:
    std::unique_ptr<Sequential> model_;           // Neural network model
    std::unique_ptr<Dataset> train_dataset_;       // Training dataset
    std::unique_ptr<Dataset> val_dataset_;        // Validation dataset
    std::unique_ptr<Optimizer> optimizer_;        // Optimizer
    std::unique_ptr<Loss> loss_fn_;               // Loss function
    std::vector<std::unique_ptr<Callback>> callbacks_;  // Training callbacks
    
    // Training parameters
    int batch_size_;                              // Batch size
    int num_epochs_;                              // Number of epochs
    bool shuffle_;                                // Whether to shuffle data
    bool verbose_;                                // Whether to print progress
    
    // Training state
    int current_epoch_;                           // Current epoch
    int current_batch_;                           // Current batch
    double current_loss_;                         // Current loss
    std::map<std::string, double> current_metrics_;  // Current metrics

public:
    /**
     * @brief Constructor for trainer
     * @param model Neural network model
     * @param train_dataset Training dataset
     * @param val_dataset Validation dataset
     * @param optimizer Optimizer
     * @param loss_fn Loss function
     * @param batch_size Batch size
     * @param num_epochs Number of epochs
     * @param shuffle Whether to shuffle data
     * @param verbose Whether to print progress
     */
    Trainer(std::unique_ptr<Sequential> model,
            std::unique_ptr<Dataset> train_dataset,
            std::unique_ptr<Dataset> val_dataset,
            std::unique_ptr<Optimizer> optimizer,
            std::unique_ptr<Loss> loss_fn,
            int batch_size = 32,
            int num_epochs = 100,
            bool shuffle = true,
            bool verbose = true);

    /**
     * @brief Destructor
     */
    ~Trainer() = default;

    // Training methods
    /**
     * @brief Train the model
     */
    void train();
    
    /**
     * @brief Train for one epoch
     * @param epoch Epoch number
     * @return Training loss and metrics
     */
    std::pair<double, std::map<std::string, double>> train_epoch(int epoch);
    
    /**
     * @brief Validate the model
     * @return Validation loss and metrics
     */
    std::pair<double, std::map<std::string, double>> validate();
    
    /**
     * @brief Train on a single batch
     * @param batch_data Batch data
     * @param batch_targets Batch targets
     * @return Batch loss
     */
    double train_batch(const Tensor& batch_data, const Tensor& batch_targets);

    // Callback management
    /**
     * @brief Add a callback
     * @param callback Unique pointer to callback
     */
    void add_callback(std::unique_ptr<Callback> callback);
    
    /**
     * @brief Remove all callbacks
     */
    void clear_callbacks();

    // Model management
    /**
     * @brief Save model to file
     * @param filepath File path to save to
     */
    void save_model(const std::string& filepath) const;
    
    /**
     * @brief Load model from file
     * @param filepath File path to load from
     */
    void load_model(const std::string& filepath);
    
    /**
     * @brief Get model reference
     * @return Reference to model
     */
    Sequential& get_model() { return *model_; }
    
    /**
     * @brief Get model reference (const version)
     * @return Const reference to model
     */
    const Sequential& get_model() const { return *model_; }

    // Training state access
    /**
     * @brief Get current epoch
     * @return Current epoch number
     */
    int get_current_epoch() const { return current_epoch_; }
    
    /**
     * @brief Get current batch
     * @return Current batch number
     */
    int get_current_batch() const { return current_batch_; }
    
    /**
     * @brief Get current loss
     * @return Current loss value
     */
    double get_current_loss() const { return current_loss_; }
    
    /**
     * @brief Get current metrics
     * @return Current metrics map
     */
    const std::map<std::string, double>& get_current_metrics() const { return current_metrics_; }

    // Training parameters
    /**
     * @brief Set batch size
     * @param batch_size New batch size
     */
    void set_batch_size(int batch_size) { batch_size_ = batch_size; }
    
    /**
     * @brief Set number of epochs
     * @param num_epochs New number of epochs
     */
    void set_num_epochs(int num_epochs) { num_epochs_ = num_epochs; }
    
    /**
     * @brief Set shuffle flag
     * @param shuffle New shuffle flag
     */
    void set_shuffle(bool shuffle) { shuffle_ = shuffle; }
    
    /**
     * @brief Set verbose flag
     * @param verbose New verbose flag
     */
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    /**
     * @brief Execute callbacks for a specific event
     * @param event Event name
     * @param args Event arguments
     */
    template<typename... Args>
    void execute_callbacks(const std::string& event, Args&&... args);
    
    /**
     * @brief Print training progress
     * @param epoch Current epoch
     * @param batch Current batch
     * @param loss Current loss
     * @param metrics Current metrics
     */
    void print_progress(int epoch, int batch, double loss, 
                       const std::map<std::string, double>& metrics) const;
    
    /**
     * @brief Print epoch summary
     * @param epoch Current epoch
     * @param train_loss Training loss
     * @param val_loss Validation loss
     * @param train_metrics Training metrics
     * @param val_metrics Validation metrics
     */
    void print_epoch_summary(int epoch, double train_loss, double val_loss,
                            const std::map<std::string, double>& train_metrics,
                            const std::map<std::string, double>& val_metrics) const;
};

} // namespace cppvision

#endif // CPPVISION_TRAINING_TRAINER_H
