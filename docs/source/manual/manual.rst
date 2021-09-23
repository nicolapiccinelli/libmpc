***************
libmpc++ manual
***************

Welcome to the manual for libmpc++, the manual is organized as follows:

* Linear MPC
* Non-linear MPC
* Tutorial

Linear MPC
==========

The linear MPC address the solution of the following convex quadratic optimization problem

.. math::
    \begin{array}{ll}
    \text{minimize}   & \sum_{k=0}^{N} (y_k-y_r)^T \Lambda_y (y_k-y_r) + \\
                      &  \sum_{k=0}^{N-1} (u_k-u_r)^T \Lambda_u (u_k-u_r) + \\
                      &  \sum_{k=0}^{N-1} (\Delta u_k - \Delta u_r)^T \Lambda_{du} (\Delta u_k - \Delta u_r) \\
    \text{subject to} & x_{k+1} = A x_k + B u_k + B^d d_k \\
                        & x_{\rm min} \le x_k  \le x_{\rm max} \\
                        & y_{\rm min} \le y_k  \le y_{\rm max} \\
                        & u_{\rm min} \le u_k  \le u_{\rm max} \\
                        & x_0 = \bar{x}
    \end{array}

where the states :math:`x_k \in \mathbf{R}^{n_x}`, the outputs :math:`y_k \in \mathbf{R}^{n_y}` and the inputs :math:`u_k \in \mathbf{R}^{n_u}` are constrained to be between some lower and upper bounds.
The problem is solved repeatedly for varying initial state :math:`\bar{x} \in \mathbf{R}^{n_x}`.

The underlying linear system used within the MPC is defined as

.. math::
    \begin{align}
        x_{k+1} &= A x_k + B u_k + B^{d} d_k\\
        y_k &= C x_k + D^{d} d_k
    \end{align}

where :math:`d_k \in \mathbf{R}^{n_{du}}` is an additional exogenous input.

Non-linear MPC
==============

The non-linear MPC address the solution of a generic non-linear optimization problem

.. math::
    \begin{array}{ll}
    \text{minimize}   & \mathcal{C}(x_k, u_k) \\
    \text{subject to} & x_{k+1} = f(x_k, u_k) \\
                        & l(x_k, y_k, u_k) \leq 0 \\
                        & h(x_k, u_k) = 0 \\
                        & x_0 = \bar{x}
    \end{array}

where the states :math:`x_k \in \mathbf{R}^{n_x}`, the outputs :math:`y_k \in \mathbf{R}^{n_y}` and the inputs :math:`u_k \in \mathbf{R}^{n_u}` can be arbitrary constrainted by defining the
function :math:`l(x_k, y_k, u_k)` and :math:`h(x_k, y_k, u_k)`. The problem is solved repeatedly for varying initial state :math:`\bar{x} \in \mathbf{R}^{n_x}` and minimizing the user defined
objective function :math:`\mathcal{C}(x_k, u_k)`.

The underlying non-linear system used within the MPC is defined as

.. math::
    \begin{align}
        x_{k+1} &= f(x_k, u_k)\\
        y_k &= g(x_k, u_k)
    \end{align}

in case of continuos time system the function :math:`\dot x = f(x, u)` should be interpreted as the vector field of the desired dynamical system.

Tutorial
========

In the following section we will present a brief introduction on how to use libmpc++, the tutorial is just to give you an overview on the main part of the library. 
Please refers to the :ref:`libmpc++ API references<libmpc++-api>` for a full list of the available functionalities.

Allocation
----------

The main classes you deal with are the API interface to the linear and non-linear MPC, called **LMPC** and **NLMPC** respectively. Depending on the allocation type desired
there are two ways of creating these objects, one by using template and one by providing the dimensions as constructor arguments. The first can be used when the dimensions of the MPC problem are
know at the compile time and the latter in the other case.

Static allocation

.. code-block:: c++

    mpc::NLMPC<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> nlmpc;
    mpc::LMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch> lmpc;

Dynamic allocation

.. code-block:: c++

    mpc::NLMPC<> nlmpc(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);
    mpc::LMPC<> lmpc(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0);

Linear MPC (LMPC)
-----------------

TBD


Non-linear MPC (LMPC)
---------------------

TBD