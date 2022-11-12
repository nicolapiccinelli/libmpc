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

Solver parametrization
----------------------

The inner solvers can be parametrized by using the following structures

Non-linear MPC solver (nlopt)

.. code-block:: c++

    NLParameters params;
        
    params.relative_ftol = 1e-10;
    params.relative_xtol = 1e-10;
    params.hard_constraints = true;

    nlmpc.setOptimizerParameters(params);

Linear MPC solver (OSQP)

.. code-block:: c++

    LParameters params;

    params.alpha = 1.6;
    params.rho = 1e-6;  
    params.eps_rel = 1e-4;
    params.eps_abs = 1e-4;   
    params.eps_prim_inf = 1e-3;
    params.eps_dual_inf = 1e-3;
    params.verbose = false;
    params.adaptive_rho = true;
    params.polish = true;    

    lmpc.setOptimizerParameters(params);

Linear MPC (LMPC)
-----------------

This example shows how to regulate a quadcopter about a reference state with constrained control input and state space

.. code-block:: c++

    lmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);

    mpc::mat<Tnx, Tnx> Ad;
    Ad << 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0,
        0.0488, 0, 0, 1, 0, 0, 0.0016, 0, 0, 0.0992, 0, 0,
        0, -0.0488, 0, 0, 1, 0, 0, -0.0016, 0, 0, 0.0992, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0992,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0.9734, 0, 0, 0, 0, 0, 0.0488, 0, 0, 0.9846, 0, 0,
        0, -0.9734, 0, 0, 0, 0, 0, -0.0488, 0, 0, 0.9846, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9846;

    mpc::mat<Tnx, Tnu> Bd;
    Bd << 0, -0.0726, 0, 0.0726,
        -0.0726, 0, 0.0726, 0,
        -0.0152, 0.0152, -0.0152, 0.0152,
        0, -0.0006, -0.0000, 0.0006,
        0.0006, 0, -0.0006, 0,
        0.0106, 0.0106, 0.0106, 0.0106,
        0, -1.4512, 0, 1.4512,
        -1.4512, 0, 1.4512, 0,
        -0.3049, 0.3049, -0.3049, 0.3049,
        0, -0.0236, 0, 0.0236,
        0.0236, 0, -0.0236, 0,
        0.2107, 0.2107, 0.2107, 0.2107;

    mpc::mat<Tny, Tnx> Cd;
    Cd.setIdentity();

    mpc::mat<Tny, Tnu> Dd;
    Dd.setZero();

    lmpc.setStateSpaceModel(Ad, Bd, Cd);

    lmpc.setDisturbances(
        mpc::mat<Tnx, Tndu>::Zero(),
        mpc::mat<Tny, Tndu>::Zero());

    mpc::cvec<Tnu> InputW, DeltaInputW;
    mpc::cvec<Tny> OutputW;

    OutputW << 0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5;
    InputW << 0.1, 0.1, 0.1, 0.1;
    DeltaInputW << 0, 0, 0, 0;

    lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, pred_hor});

    mpc::cvec<Tnx> xmin, xmax;
    xmin << -M_PI / 6, -M_PI / 6, -mpc::inf, -mpc::inf, -mpc::inf, -1,
        -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf;

    xmax << M_PI / 6, M_PI / 6, mpc::inf, mpc::inf, mpc::inf, mpc::inf,
        mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf;

    mpc::cvec<Tny> ymin, ymax;
    ymin.setOnes();
    ymin *= -mpc::inf;
    ymax.setOnes();
    ymax *= mpc::inf;

    mpc::cvec<Tnu> umin, umax;
    double u0 = 10.5916;
    umin << 9.6, 9.6, 9.6, 9.6;
    umin.array() -= u0;
    umax << 13, 13, 13, 13;
    umax.array() -= u0;

    lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, pred_hor});

    mpc::cvec<Tny> yRef;
    yRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    lmpc.setReferences(yRef, mpc::cvec<Tnu>::Zero(), mpc::cvec<Tnu>::Zero(), {0, pred_hor});

    auto res = lmpc.step(mpc::cvec<Tnx>::Zero(), mpc::cvec<Tnu>::Zero());
    lmpc.getOptimalSequence();

Non-linear MPC (LMPC)
---------------------

This example shows how to drives the states of a Van der Pol oscillator to zero with constrained control input

.. code-block:: c++

    double ts = 0.1;

    nlmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    nlmpc.setContinuosTimeModel(ts);

    auto stateEq = [&](mpc::cvec<Tnx>& dx,
                       mpc::cvec<Tnx> x,
                       mpc::cvec<Tnu> u) {
        dx(0) = ((1.0 - (x(1) * x(1))) * x(0)) - x(1) + u(0);
        dx(1) = x(0);
    };

    nlmpc.setStateSpaceFunction(stateEq);

    nlmpc.setObjectiveFunction([&](mpc::mat<Tph + 1, Tnx> x,
                                   mpc::mat<Tph + 1, Tnu> u,
                                   double) {
        return x.array().square().sum() + u.array().square().sum();
    });

    nlmpc.setIneqConFunction([&](mpc::cvec<ineq_c>& in_con,
                                 mpc::mat<Tph + 1, Tnx>,
                                 mpc::mat<Tph + 1, Tny>,
                                 mpc::mat<Tph + 1, Tnu> u,
                                 double) {
        for (int i = 0; i < ineq_c; i++) {
            in_con(i) = u(i, 0) - 0.5;
        }
    });

    mpc::cvec<Tnx> modelX, modeldX;

    modelX(0) = 0;
    modelX(1) = 1.0;

    auto r = nlmpc.getLastResult();

    for (;;) {
        r = nlmpc.step(modelX, r.cmd);
        auto seq = nlmpc.getOptimalSequence();
        stateEq(modeldX, modelX, r.cmd);
        modelX += modeldX * ts;

        if (std::fabs(modelX[0]) <= 1e-2 && std::fabs(modelX[1]) <= 1e-1) {
            break;
        }
    }