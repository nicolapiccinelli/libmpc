.. _libmpc++-api:

**************
API references
**************

The libmpc++ API consists of the following parts:

* :ref:`IMPC.hpp <impc-api>`: Common MPC interface
* :ref:`LMPC.hpp <linear-mpc-api>`: Linear MPC interface
* :ref:`NLMPC.hpp <nonlinear-mpc-api>`: Non-linear MPC interface
* :ref:`MPPI <mppi-api>`: Sampling-based optimizer for non-linear MPC
* :ref:`Integrator.hpp <integrator-api>`: Numerical integrators
* :ref:`Utils.hpp <utils-api>`: Utilities

.. _impc-api:

The MPC interface
=================

.. doxygenclass:: mpc::IMPC
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

.. _linear-mpc-api:

The linear MPC interface
========================

.. doxygenclass:: mpc::LMPC
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

.. _nonlinear-mpc-api:

The non-linear MPC interface
============================

.. doxygenclass:: mpc::NLMPC
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

The `mpc::NLMPC` interface supports multiple internal optimization backends. In addition to the default gradient-based solver (NLopt, e.g., SLSQP), you can select the sampling-based `OptimizerType::MPPI` which implements Model Predictive Path Integral control. See the section below for MPPI references and parameters.

.. _integrator-api:

Numerical integrators
=====================

.. doxygenclass:: mpc::RK4
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

.. _mppi-api:

Model Predictive Path Integral (MPPI)
=====================================

The MPPI optimizer is available for non-linear MPC and provides a gradient-free, sampling-based alternative to the default NLopt backend. Use `OptimizerType::MPPI` when constructing `mpc::NLMPC` or via the appropriate constructor overload.

.. doxygenclass:: mpc::MPPIOptimizer
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

MPPI parameters
---------------

.. doxygenstruct:: mpc::MPPIParameters
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

.. _utils-api:

Utilities
=========

.. doxygenfile:: Utils.hpp
    :project: libmpc++
    :path: ../web/doxygen-build/xml
