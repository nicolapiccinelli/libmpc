# libmpc++
libmpc++ is a C++ library to solve linear and non-linear MPC. The library requires at least C++17 and it
is tested to work on Linux, macOS and Windows.

The libmpc++ website can be found at the following link: https://altairlab.gitlab.io/libmpc/
## Dependecies
Deploy:
- `Eigen3` (https://eigen.tuxfamily.org/index.php?title=Main_Page)
- `OSQP` for linear MPC (https://osqp.org)
- `NLopt` for non-linear MPC (https://nlopt.readthedocs.io/en/latest/)

Development:
- `boost` for stacktrace debug (https://www.boost.org)
- `Catch2` test suite (https://github.com/catchorg/Catch2)
- `gcovr`, `lcov` GCC code coverage

## Usage
The library is header only, simply add the include folder to your project and install the dependecies listed above