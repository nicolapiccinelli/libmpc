cmake_minimum_required(VERSION 3.12)
project(pympcxx LANGUAGES CXX)

# Find Pybind11
add_subdirectory(deps/pybind11)
find_package(mpc++ CONFIG REQUIRED)

# Add your source files here
pybind11_add_module(${PROJECT_NAME} MODULE pybind_export.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE deps/pybind11/include)
target_link_libraries(${PROJECT_NAME} PRIVATE mpc++)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)