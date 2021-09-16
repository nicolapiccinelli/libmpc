# libmpc++
References link: https://altairlab.gitlab.io/libmpc/
## Dependecies:
Deploy:
- `eigen`
- `osqp` for linear MPC
- `nlopt` for non-linear MPC

Development:
- `boost` for stacktrace debug
- `Catch2` test suite
- `gcovr`, `lcov` GCC code coverage

## Usage
The library is header only, simply add the include folder to your project. The project use static allocation only, big optimization problem can't be currently handled.
