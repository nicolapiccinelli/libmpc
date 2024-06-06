# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.0)

# Set the project name
project(examples)

# Enable C++20
set(CMAKE_CXX_STANDARD 20)
# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# Disable eigen stack allocation warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_STACK_ALLOCATION_LIMIT=0")
# Enable openmp

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Find the mpc++ package
find_package(mpc++ CONFIG REQUIRED)

# Include the mpc++ headers
include_directories(${mpc++_INCLUDE_DIRS})

# Get all the .cpp files in the directory
file(GLOB CPP_FILES *.cpp)

# Put the executables in the bin directory in the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Create a target for each .cpp file
foreach(CPP_FILE ${CPP_FILES})
    # Get the file name without extension
    get_filename_component(TARGET_NAME ${CPP_FILE} NAME_WE)

    # Write the file name to the console
    message(STATUS "Adding target for ${CPP_FILE} -> ${TARGET_NAME}")

    # Add the target
    add_executable(${TARGET_NAME} ${CPP_FILE})
    target_link_libraries(${TARGET_NAME} mpc++)
endforeach()