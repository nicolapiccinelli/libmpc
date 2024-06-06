****************
libmpc++ library
****************

Introduction
============

**libmpc++** is a free/open-source library for solving **linear** and **non-linear** Model Predictive Control (MPC). The library is
written in standard C++20 and provides static and dynamic memory allocation via templated interfacing classes. It is available on
Linux, MacOs and Windows and comes with a limited set of dependecies. It provides:

* Support for linear and non-linear MPC optimal control problem formulation
* Handles discrete-time and continuous-time (for the non-linear MPC only) system's dynamics definition
* Different length for the prediction and control horizon
* Automatic Jacobian approximation (using the trapezoidal rule) for non-linear MPC
* Header-only implementation
* Free/open-source software

The library depends on the following external libraries which must be installed on the machine before using libmpc++

* *Eigen3* header-only linear algebra library (https://eigen.tuxfamily.org/index.php?title=Main_Page)
* *NLopt* set of solvers for nonlinear programming (https://nlopt.readthedocs.io/en/latest/)
* *OSQP* solver for convex quadratic programming (https://osqp.org)

Download and installation
=========================

The latest version of libmpc++ is available from GitHub https://github.com/nicolapiccinelli/libmpc/releases and does not require any
installation process other than the one required by its dependecies.

Documentation
=============

See the :ref:`libmpc++ API references<libmpc++-api>` for information on how to use this library and please :ref:`cite libmpc++<libmpc++-cite>` 
and the authors of the algorithm(s) you use in any publication that stems from your use of libmpc++.

Feedback (closed loop development)
==================================

For bug reports and feature requests, please refers to the github issue https://github.com/nicolapiccinelli/libmpc/issues