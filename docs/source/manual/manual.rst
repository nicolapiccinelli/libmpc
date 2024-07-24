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
                        & s_{\rm min} \le x_s^T x_k + u_s^T u_k \le s_{\rm max}\\
                        & x_0 = \bar{x}
    \end{array}

where the states :math:`x_k \in \mathbf{R}^{n_x}`, the outputs :math:`y_k \in \mathbf{R}^{n_y}` and the inputs :math:`u_k \in \mathbf{R}^{n_u}` are constrained to be between some lower and upper bounds.
THe states and the inputs can be also subjected to the so called "scalar constraints", where :math:`x_s` and :math:`u_s` are constant vectors. 
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

in case of continuous time system the function :math:`\dot x = f(x, u)` should be interpreted as the vector field of the desired dynamical system. 
The dynamical system is transformed into discrete-time using the trapezoidal rule internally.

Tutorial
========

In the following section we will present a brief introduction on how to use libmpc++, the tutorial is just to give you an overview on the main part of the library. 
Please refers to the :ref:`libmpc++ API references<libmpc++-api>` for a full list of the available functionalities.

Allocation
----------

The main classes you deal with are the API interface to the linear and non-linear MPC, called **LMPC** and **NLMPC** respectively. 
Depending on the allocation type desired there are two ways of creating these objects, one by using template and one by providing the dimensions as constructor arguments. 
The first can be used when the dimensions of the MPC problem are know at the compile time and the latter in the other case.

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
    params.absolute_ftol = 1e-10;
    params.absolute_xtol = 1e-10;
    params.time_limit = 0;
    
    params.hard_constraints = true;
    params.enable_warm_start = false;

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
    params.time_limit = 0;
    params.enable_warm_start = false;
    params.verbose = false;
    params.adaptive_rho = true;
    params.polish = true;

    lmpc.setOptimizerParameters(params);

Optimization result
-------------------

The optimization result is stored in the **Result** structure. The structure contains the following fields:

* solver_status: the return code of the optimization solver
* is_feasible: a boolean flag that indicates if the optimal solution is feasible
* solver_status_msg: the status message of the optimization solver
* cost: the optimal cost of the optimization problem
* status: the status of the MPC
* cmd: the optimal control input

.. code-block:: c++

    Result<Tnu> res = ctrl.optimize(mpc::cvec<Tnx>::Zero(), mpc::cvec<Tnu>::Zero());

    std::cout << "Solver status code: " << res.solver_status << std::endl;
    std::cout << "Is feasible: " << res.is_feasible << std::endl;
    std::cout << "Solver status message: " << res.solver_status_msg << std::endl;
    std::cout << "Cost: " << res.cost << std::endl;
    std::cout << "Status: " << res.status << std::endl;
    std::cout << "Control input: " << res.cmd.transpose() << std::endl;

The return code of the optimization solver changes depending on the solver used and it is described when possible
by the status message. To have a coherent status code of the optimization problem the **status** field is used. 
The status can be one of the following:

* SUCCESS (0): the optimization problem has been solved
* MAX_ITERATION (1): the optimization problem has reached the maximum number of iterations or the maximum time limit
* INFEASIBLE (2): the optimization problem is infeasible
* ERROR (3): an error occurred during the optimization
* UNKNOWN (4): the status of the optimization problem is unset or unknown

.. code-block:: c++
    enum ResultStatus
        {
            SUCCESS,
            MAX_ITERATION,
            INFEASIBLE,
            ERROR,
            UNKNOWN
        };

If needed the optimal sequence along the prediction horizon can be retrieved by calling the **getOptimalSequence** method.
The sequence contains also the initial condition of the optimization problem.

.. code-block:: c++

    OptSequence<Tnx, Tny, Tnu, Tph + 1> seq = ctrl.getOptimalSequence();
    
    std::cout << "Optimal sequence: " << std::endl;
    std::cout << seq.state << std::endl;
    std::cout << seq.input << std::endl;
    std::cout << seq.output << std::endl;

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

    lmpc.setStateBounds(xmin, xmax, {0, pred_hor});
    lmpc.setInputBounds(umin, umax, {0, pred_hor});
    lmpc.setOutputBounds(ymin, ymax, {0, pred_hor});

    mpc::cvec<Tny> yRef;
    yRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    lmpc.setReferences(yRef, mpc::cvec<Tnu>::Zero(), mpc::cvec<Tnu>::Zero(), {0, pred_hor});

    auto res = lmpc.optimize(mpc::cvec<Tnx>::Zero(), mpc::cvec<Tnu>::Zero());
    lmpc.getOptimalSequence();

Non-linear MPC (LMPC)
---------------------

This example shows how to drives the states of a Van der Pol oscillator to zero with constrained control input

.. code-block:: c++

    double ts = 0.1;

    nlmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    nlmpc.setDiscretizationSamplingTime(ts);

    auto stateEq = [&](mpc::cvec<Tnx>& dx,
                       const mpc::cvec<Tnx>& x,
                       const mpc::cvec<Tnu>& u) {
        dx(0) = ((1.0 - (x(1) * x(1))) * x(0)) - x(1) + u(0);
        dx(1) = x(0);
    };

    nlmpc.setStateSpaceFunction([&](mpc::cvec<Tnx> &dx,
                                    const mpc::cvec<Tnx>& x,
                                    const mpc::cvec<Tnu>& u,
                                    const unsigned int&)
                                    { stateEq(dx, x, u); });

    nlmpc.setObjectiveFunction([&](const mpc::mat<Tph + 1, Tnx>& x,
                                   const mpc::mat<Tph + 1, Tny>& y,
                                   const mpc::mat<Tph + 1, Tnu>& u,
                                   const double&) {
        return x.array().square().sum() + u.array().square().sum();
    });

    nlmpc.setIneqConFunction([&](mpc::cvec<ineq_c>& in_con,
                                 const mpc::mat<Tph + 1, Tnx>&,
                                 const mpc::mat<Tph + 1, Tny>&,
                                 const mpc::mat<Tph + 1, Tnu>& u,
                                 const double&) {
        for (int i = 0; i < ineq_c; i++) {
            in_con(i) = u(i, 0) - 0.5;
        }
    });

    mpc::cvec<Tnx> modelX, modeldX;

    modelX(0) = 0;
    modelX(1) = 1.0;

    auto r = nlmpc.getLastResult();

    for (;;) {
        r = nlmpc.optimize(modelX, r.cmd);
        auto seq = nlmpc.getOptimalSequence();
        stateEq(modeldX, modelX, r.cmd);
        modelX += modeldX * ts;

        if (std::fabs(modelX[0]) <= 1e-2 && std::fabs(modelX[1]) <= 1e-1) {
            break;
        }
    }

Import libmpc++ in your project
-------------------------------

libmpc++ is a header only library, so you can just copy the content of the include folder in your project. 
If your project uses CMake, you can import libmpc++ as a package with the following commands:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.0)
    project(your_project_name)

    # set the C++ standard to C++ 20
    set(CMAKE_CXX_STANDARD 20)
    # set the C++ compiler to use O3
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

    find_package(mpc++ CONFIG REQUIRED)

    # # Declare a C++ library
    include_directories(${mpc++_INCLUDE_DIRS})
    add_executable(${PROJECT_NAME} main.cpp)
    target_link_libraries(${PROJECT_NAME} mpc++)