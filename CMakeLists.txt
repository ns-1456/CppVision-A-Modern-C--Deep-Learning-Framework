cmake_minimum_required(VERSION 3.15)
project(mnist_cnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build tests" OFF)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES
    src/*.cpp
)

add_executable(mnist_cnn ${SRC_FILES}
        src/data/mnist_reader.cpp
        src/data/mnist_reader.h
        src/data/data_processor.cpp
        src/data/data_processor.h)

# If you later add tests, you can enable them here
# enable_testing()
# add_subdirectory(tests)
