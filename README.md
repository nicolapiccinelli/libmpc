# libmpc++
libmpc++ is a C++ library to solve linear and non-linear MPC. The library is written in modern **C++20** and it
is tested to work on Linux, macOS and Windows.

* gcc version (>= 10.3.0)

The libmpc++ website can be found at the following link: https://altairlab.gitlab.io/optcontrol/libmpc/

## Dependecies
The library depends on the following external libraries which must be installed on the machine before using libmpc++

* *Eigen3* header-only linear algebra library (https://eigen.tuxfamily.org/index.php?title=Main_Page)
* *NLopt* set of solvers for nonlinear programming (https://nlopt.readthedocs.io/en/latest/)
* *OSQP* solver for convex quadratic programming (https://osqp.org)

If you are a developer, to setup the debug environment you also need to install:
- `boost` for stacktrace debug (https://www.boost.org)
- `Catch2` test suite (https://github.com/catchorg/Catch2)
- `gcovr`, `lcov` GCC code coverage

## Usage
The latest version of libmpc++ is available from GitHub https://github.com/nicolapiccinelli/libmpc/releases and does not require any
installation process other than the one required by its dependecies.
