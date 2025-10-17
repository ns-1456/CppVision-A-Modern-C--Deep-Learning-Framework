cmake_minimum_required(VERSION 3.15)
project(cppvision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CppVision Library Sources
set(CPPVISION_CORE_SOURCES
    src/cppvision/core/tensor.cpp
)

set(CPPVISION_LAYER_SOURCES
    src/cppvision/layers/dense.cpp
    src/cppvision/layers/conv2d.cpp
    src/cppvision/layers/maxpool2d.cpp
    src/cppvision/layers/activations.cpp
)

set(CPPVISION_MODEL_SOURCES
    src/cppvision/models/sequential.cpp
)

set(CPPVISION_DATASET_SOURCES
    # src/cppvision/datasets/mnist_dataset.cpp  # Temporarily disabled due to compilation issues
)

set(CPPVISION_TRAINING_SOURCES
    src/cppvision/training/loss.cpp
    src/cppvision/training/metrics.cpp
    # src/cppvision/training/trainer.cpp  # Temporarily disabled due to compilation issues
)

set(CPPVISION_OPTIMIZER_SOURCES
    src/cppvision/optimizers/sgd.cpp
    src/cppvision/optimizers/adam.cpp
)

# Legacy sources (existing MNIST implementation) - REMOVED
# set(LEGACY_SOURCES
#     src/data/mnist_reader.cpp
#     src/data/data_processor.cpp
#     src/utils/math_utils.cpp
# )

# Combine all CppVision sources
set(CPPVISION_ALL_SOURCES
    ${CPPVISION_CORE_SOURCES}
    ${CPPVISION_LAYER_SOURCES}
    ${CPPVISION_MODEL_SOURCES}
    ${CPPVISION_DATASET_SOURCES}
    ${CPPVISION_TRAINING_SOURCES}
    ${CPPVISION_OPTIMIZER_SOURCES}
)

# Create CppVision static library
add_library(cppvision STATIC ${CPPVISION_ALL_SOURCES})

# Set library properties
set_target_properties(cppvision PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Main executable
add_executable(mnist_cnn src/main.cpp)
target_link_libraries(mnist_cnn cppvision)

# Legacy executable (if needed) - REMOVED
# add_executable(mnist_legacy ${LEGACY_SOURCES} src/main_legacy.cpp)
# target_link_libraries(mnist_legacy)

# Example executable (if BUILD_EXAMPLES is ON) - DISABLED
# if(BUILD_EXAMPLES)
#     add_executable(cppvision_example examples/basic_cnn.cpp)
#     target_link_libraries(cppvision_example cppvision)
# endif()

# If you later add tests, you can enable them here
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print build information
message(STATUS "CppVision Framework Build Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  CppVision Sources: ${CPPVISION_ALL_SOURCES}")
