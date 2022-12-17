#!bin/bash
cd "$(dirname "$0")"

# Create build directory
mkdir -p build && cd build

# Build Library
cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_EXPORT_COMPILE_COMMANDS=ON .. 
make -j$(($(nproc)-1))
make test 
make install