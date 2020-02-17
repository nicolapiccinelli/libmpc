cmake_minimum_required( VERSION 3.2 )

## Specify a project name
project(mpc)

set(CMAKE_BUILD_TYPE RELEASE)

## C++ compiler options
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++1z -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -rdynamic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Load CMAKE configuration from environment variables
set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})                      

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    include_directories(${PROJECT_SOURCE_DIR}/deps/stacktrace)
    add_definitions(-Ddebug)
endif()

if(APPLE)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib/)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

## Find all source files to be compiled into object files
file(GLOB SOURCES src/*.cpp)

add_executable(vanderpol test/vanderpol.cpp)
target_link_libraries(vanderpol nlopt m)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    target_link_libraries(vanderpol ${PROJECT_SOURCE_DIR}/deps/stacktrace/libstacktrace.so)
endif()