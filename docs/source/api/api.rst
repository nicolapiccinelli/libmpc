**************
API References
**************

The libmpc API consists of the following parts:

* :ref:`LMPC.hpp <linear-mpc-api>`: Linear MPC control interface
* :ref:`NLMPC.hpp <nonlinear-mpc-api>`: Non-linear MPC control interface
* :ref:`Integrator.hpp <integrator-api>`: Numerical integrators
* :ref:`Utils.hpp <utils-api>`: Generic utilities functions

.. _linear-mpc-api:

The linear MPC control interface
================================

.. doxygenclass:: mpc::LMPC
    :project: libmpc
    :path: ../web/doxygen-build/xml
    :members:

.. _nonlinear-mpc-api:

The non-linear MPC control interface
====================================

.. doxygenclass:: mpc::NLMPC
    :project: libmpc
    :path: ../web/doxygen-build/xml
    :members:

.. _integrator-api:

Numerical integrators
=====================

.. doxygenclass:: mpc::RK4
    :project: libmpc
    :path: ../web/doxygen-build/xml
    :members:

.. _utils-api:

Generic utilities functions
===========================

.. doxygenfile:: Utils.hpp
    :project: libmpc
    :path: ../web/doxygen-build/xml
