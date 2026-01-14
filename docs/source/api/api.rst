.. _libmpc++-api:

**************
API references
**************

The libmpc++ API consists of the following parts:

* :ref:`IMPC <impc-api>`: Common MPC interface
* :ref:`LMPC <linear-mpc-api>`: Linear MPC interface
* :ref:`NLMPC <nonlinear-mpc-api>`: Non-linear MPC interface
* :ref:`Parameters <parameters_mpc>`: MPC parameters structures
* :ref:`Integrator <integrator-api>`: Numerical integrators for system dynamics
* :ref:`Utils <utils-api>`: Utilities and helper functions

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

.. _parameters_mpc:

Linear MPC parameters
----------------------
.. doxygenstruct:: mpc::LMPCParameters
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

Non-linear MPC parameters
-------------------------
.. doxygenstruct:: mpc::NLMPCParameters
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

MPPI parameters
---------------

.. doxygenstruct:: mpc::MPPIParameters
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

.. _integrator-api:

Numerical integrators
=====================

.. doxygenclass:: mpc::RK4
    :project: libmpc++
    :path: ../web/doxygen-build/xml
    :members:

.. _utils-api:

Utilities
=========

.. doxygenfile:: Utils.hpp
    :project: libmpc++
    :path: ../web/doxygen-build/xml
