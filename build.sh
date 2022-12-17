#!bin/bash
cd "$(dirname "$0")"

# Create build directory
mkdir -p build && cd build

# Build Library
cmake -D CMAKE_BUILD_TYPE=Release .. 
make
sudo make install