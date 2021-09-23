.. _libmpc++-api:

**************
API references
**************

The libmpc++ API consists of the following parts:

* :ref:`IMPC.hpp <impc-api>`: Common MPC interface
* :ref:`LMPC.hpp <linear-mpc-api>`: Linear MPC interface
* :ref:`NLMPC.hpp <nonlinear-mpc-api>`: Non-linear MPC interface
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
