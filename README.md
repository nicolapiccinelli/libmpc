# libmpc++
libmpc++ is a C++ library to easily solve linear and non-linear MPC, the library requires at least C++17 and it
is tested on Linux, MacOs and Windows.

For the full references please referes to the following link: https://altairlab.gitlab.io/libmpc/
## Dependecies
Deploy:
- `eigen`
- `osqp` for linear MPC
- `nlopt` for non-linear MPC

Development:
- `boost` for stacktrace debug
- `Catch2` test suite
- `gcovr`, `lcov` GCC code coverage

## Usage
The library is header only, simply add the include folder to your project and install the dependecies listed above