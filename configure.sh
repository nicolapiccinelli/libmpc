#!/usr/bin/env bash

# read the parameter --disable-test from the command line 
# to choose if you want to disable the test suite installation
# default is to install the test suite
while test $# -gt 0; do
  case "$1" in
    --disable-test)
      shift
      DISABLE_TEST=1
      ;;
    *)
      break
      ;;
  esac
done

# install eigen3 from apt
apt-get install -y libeigen3-dev

# folder for dependencies from source
mkdir -p deps
cd deps

# install nlopt from master branch
git clone https://github.com/stevengj/nlopt.git
cd nlopt
mkdir build
cd build
cmake -D NLOPT_PYTHON=OFF -D NLOPT_OCTAVE=OFF -D NLOPT_MATLAB=OFF -D NLOPT_GUILE=OFF -D NLOPT_SWIG=OFF ..
make -j
make install

# install osqp from master branch
cd ../../
git clone --recursive https://github.com/osqp/osqp
cd osqp
git checkout v0.6.3
git submodule update --init --recursive
mkdir build
cd build
cmake -G "Unix Makefiles" ..
cmake --build .
cmake --build . --target install

#########################
# this is the test suite
#########################

if [ -z ${DISABLE_TEST+x} ]; then
    # install catch2 from master branch >= 3.x
    cd ../../
    git clone https://github.com/catchorg/Catch2.git
    cd Catch2
    mkdir build
    cd build
    cmake ..
    make -j
    make install
fi