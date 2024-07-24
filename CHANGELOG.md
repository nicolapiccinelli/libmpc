# Changelog

## [0.6.2] - 2024-07-24
### Added
- The `Result` struct now contains the feasibility of the solution vector in the `is_feasible` field

### Fixed
- Version 0.6 and 0.6.1 were affected by a bug which was preventing the proper computation of the Jacobian of the user-defined inequality constraints. This bug has been fixed in this version

## [0.6.1] - 2024-06-07
### Fixed
- Fixed the horizon slicing for the non-linear mpc. The horizon slicing was not working properly when set via the HorizonSlice struct

## [0.6.0] - 2024-06-05
### Added
- Added support for input and state bound constraints in the non-linear mpc. These constraints are actually restricing the search space of the optimization problem
and are obeyed by the solver also during intermediate steps of the optimization problem
- Added warm start support in the non-linear mpc. The warm start is disabled by default and can be enabled using the parameter `enable_warm_start`

### Changed
- Breaking change: The stopping criterias in the non-linear mpc parameters are now disabled by default. The only enabled criteria is the maximum number of iterations
- Breaking change: The optimal sequence returned by the linear and non-linear mpc is now containing also the initial condition
- The Jacobians of the constraints in the non-linear mpc are now estimated using the trapeizoial rule
- The functions to set the bound constraints now uses a dedicated structure to define the horizon span
- Breaking change: The functions setConstraints are now split in setStateBounds, setInputBounds and setOutputBounds
- Breaking change: The fields retcode and status_msg in the result struct of the non-linear mpc are now solver_status and solver_status_msg respectively

## [0.5.0] - 2024-05-17
### Added
- Python bindings for the library using pybind11 (pympcxx)
- The result struct now contains a string to describe the status of the optimization problem
- Added new parameters for the nonlinear mpc (time_limit, absolute_ftol, absolute_xtol)

### Changed
- Breaking change: some of the APIs have been refactored. New APIs: setDiscretizationSamplingTime, setExogenousInputs, optimize
substitute setContinuosTimeModel, setExogenuosInputs, step
- Improved error handling in the NLopt interface

### Fixed
- The set of the discretization sampling time was not working properly in the non-linear mpc

## [0.4.2] - 2024-01-14
### Added
- Added examples to show how to use the library

### Fixed
- Fixed the cmake target configuration to properly target the library

## [0.4.1] - 2024-01-30
### Changed
- Removed coloured output for the integrated logger
- Configure script now can be used to avoid the installation of the test suite

### Fixed
- Fixed dockerfile to use the 0.6.3 version of the OSQP solver

## [0.4.0] - 2023-05-20

### Added
- Added profiler to measure statistics of the optimization problem
- In linear mpc is now possible to override the warm start of the optimization problem
- Added support for OSQP warm start in linear mpc

### Changed
- The linear mpc parameters now allows to enable the warm start of the optimization problem
- The mpc result structure now contains a status field to check if the optimization problem has been solved
- CMakelists.txt has been refactored to export the INCLUDE_DIRS variable

## [0.3.1] - 2023-03-06

### Fixed
- The computation of the scalar multipler was not correct

## [0.3.0] - 2023-03-04

### Added
- Added new api in linear mpc to add a scalar constraints

### Changed
- In linear mpc the last input command is now used to initialize the optimal control problem

### Fixed
- The default value for the box constraints in linear mpc are now set -inf and inf
- In linear mpc optimal input sequence was erroneously the delta input sequence

## [0.2.0] - 2023-01-09

### Added
- Improved performances of non-linear mpc
### Changed
- Added support in non-linear for output penalization in objective function (Breaking change)
- In non-linear prediction step index is now available in the model update function (Breaking change)

## [0.1.0] - 2022-11-12
### Added
- Added support in linear mpc to define the references, weights, constraints and exogenous inputs different in each prediction step
- Added general support to the retrival of the optimal sequence (state, input and output)

### Changed
- The API to set the references, weights constraints and exogenous inputs using vector now requires a span of the horizon (Breaking changes)
- Added new APIs to define the references, weights, constraints and exogenous inputs matrices to the whole horizon
- Internal structure of the library has been refactored to separate non-linear and linear classes

## [0.0.9] - 2022-07-06
### Added
- Added support for multiple mpc instances in the same executable

### Changed
- Minimum c++ version required updated to c++20

## [0.0.8.2] - 2022-06-20

### Fixed
- Fixed issue on compilation

## [0.0.8.1] - 2022-06-15

### Changed
- Catch2 dependency updated to version 3

### Fixed
- Improved performances while building the optimization problem in LOptimizer

## [0.0.8] - 2022-03-02
### Added
- Changelog added :)
### Fixed
- Fixed a memory leak in LOptimizer
- Fixed the parameters handling in LOptimizer
