/*
 * Copyright (c) 2023-2026 Nicola Piccinelli
 * All rights reserved.
 *
 * This file is an example of a 6-DOF variable impedance controller
 * for circular trajectory tracking, inspired by the user's provided code.
 */
#include <mpc/NLMPC.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

// Use MPPI as the optimizer
#define MPPI_ENABLED

int main()
{
    // --- 1. Define System Dimensions (6-DOF) ---
    constexpr int num_dof = 6;
    // state x = [e, e_dot, m_diag, d_diag, k_diag]
    constexpr int num_states = num_dof * 5; // 6*5 = 30
    // input u = [m_dot, d_dot, k_dot]
    constexpr int num_inputs = num_dof * 3; // 6*3 = 18
    // output y = x
    constexpr int num_output = num_states; // 30

    // --- 2. Define MPC Horizons ---
    constexpr int pred_hor = 10;
    constexpr int ctrl_hor = 10;

    // --- 3. Define Constraint Dimensions ---
    constexpr int ineq_c = 0;
    constexpr int eq_c = 0;

    // Discretization time
    double ts = 0.05;

    // --- 5. Configure Optimizer (MPPI) ---
#ifdef MPPI_ENABLED
    mpc::NLMPC<
        num_states, num_inputs, num_output,
        pred_hor, ctrl_hor,
        ineq_c, eq_c>
        controller(mpc::OptimizerType::MPPI);

    mpc::MPPIParameters params;
    params.maximum_iteration = 1; // 1 is common for MPPI
    params.num_rollouts = 2048;

    params.sigma = mpc::cvec<num_inputs>();
    params.sigma << 
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        15.0, 15.0, 15.0, 15.0, 15.0, 15.0;

    params.lambda = 0.1;
    params.enable_warm_start = true;
    params.state_bound_penalty = 1000.0;
    params.ineq_penalty = 0.0;
    params.barrier_steepness = 100.0;
    params.eq_penalty = 0;
    params.beta = 1.0;
    params.rho = 0.05;
    params.integration_substeps = 1;
    controller.setOptimizerParameters(params);
#else
    mpc::NLMPC<
        num_states, num_inputs, num_output,
        pred_hor, ctrl_hor,
        ineq_c, eq_c>
        controller;

    mpc::NLParameters params;
    params.maximum_iteration = 1000;
    params.time_limit = 0.1; // [s]
    controller.setOptimizerParameters(params);
#endif

    controller.setLoggerLevel(mpc::Logger::LogLevel::NORMAL);
    controller.setDiscretizationSamplingTime(ts);

    // --- 6. Define Simulation & Cost Variables ---

    // Simulated constant external wrench (force/torque)
    // Constant -0.5N force on Z-axis
    mpc::cvec<num_dof> external_wrench;
    external_wrench.setZero();
    external_wrench(2) = -10.5;

    // Cost function weighting matrices
    // Q for state x = [e, e_dot, m, d, k]
    mpc::mat<num_states, num_states> Q;
    Q.setZero();
    // High weight on tracking error (e, e_dot)
    Q.diagonal().segment<num_dof * 2>(0).setConstant(100.0);
    // Low weight on impedance values (m, d, k)
    Q.diagonal().segment<num_dof * 3>(num_dof * 2).setConstant(0.01);

    // R for input u = [m_dot, d_dot, k_dot]
    mpc::mat<num_inputs, num_inputs> R;
    R.setIdentity();
    R.diagonal() *= 1.0;

    // --- 7. Define Trajectory and State-Space Dynamics ---

    // Trajectory parameters
    double traj_radius = 0.5; // [m]
    double traj_omega = 0.5;  // [rad/s]

    // Helper function to get trajectory at a specific time
    auto get_trajectory = [&](double t,
                              mpc::cvec<num_dof> &p_des,
                              mpc::cvec<num_dof> &p_dot_des,
                              mpc::cvec<num_dof> &p_ddot_des)
    {
        p_des.setZero();
        p_dot_des.setZero();
        p_ddot_des.setZero();

        double angle = traj_omega * t;
        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);

        // Position
        p_des(0) = traj_radius * cos_a; // X
        p_des(1) = traj_radius * sin_a; // Y

        // Velocity
        p_dot_des(0) = -traj_radius * traj_omega * sin_a; // Vx
        p_dot_des(1) = traj_radius * traj_omega * cos_a;  // Vy

        // Acceleration
        p_ddot_des(0) = -traj_radius * traj_omega * traj_omega * cos_a; // Ax
        p_ddot_des(1) = -traj_radius * traj_omega * traj_omega * sin_a; // Ay
    };

    // This function defines the system's *error* evolution: de/dt = f(e, u)
    // The MPC model is *only* the error dynamics (a regulator)
    auto stateEq = [&](
                       mpc::cvec<num_states> &dx,
                       const mpc::cvec<num_states> &x,
                       const mpc::cvec<num_inputs> &u,
                       const unsigned int &) // horizon step (unused but part of signature)
    {
        // Extract error states
        mpc::cvec<num_dof> e = x.segment<num_dof>(0);
        mpc::cvec<num_dof> e_dot = x.segment<num_dof>(num_dof);

        // Extract impedance states
        mpc::cvec<num_dof> m_diag = x.segment<num_dof>(num_dof * 2);
        mpc::cvec<num_dof> d_diag = x.segment<num_dof>(num_dof * 3);
        mpc::cvec<num_dof> k_diag = x.segment<num_dof>(num_dof * 4);

        // Extract inputs (derivatives)
        mpc::cvec<num_dof> m_dot = u.segment<num_dof>(0);
        mpc::cvec<num_dof> d_dot = u.segment<num_dof>(num_dof);
        mpc::cvec<num_dof> k_dot = u.segment<num_dof>(num_dof * 2);

        // Form diagonal impedance matrices
        mpc::mat<num_dof, num_dof> M = m_diag.asDiagonal();
        mpc::mat<num_dof, num_dof> D = d_diag.asDiagonal();
        mpc::mat<num_dof, num_dof> K = k_diag.asDiagonal();

        // System dynamics (error dynamics)
        dx.segment<num_dof>(0) = e_dot;
        dx.segment<num_dof>(num_dof) = M.inverse() * (external_wrench - K * e - D * e_dot);
        dx.segment<num_dof>(num_dof * 2) = m_dot;
        dx.segment<num_dof>(num_dof * 3) = d_dot;
        dx.segment<num_dof>(num_dof * 4) = k_dot;
    };

    controller.setStateSpaceFunction(stateEq);

    // --- 8. Define Objective Function ---
    controller.setObjectiveFunction([&](
                                        const mpc::mat<pred_hor + 1, num_states> &x,
                                        const mpc::mat<pred_hor + 1, num_output> &,
                                        const mpc::mat<pred_hor + 1, num_inputs> &u,
                                        double)
                                    {
        double cost = 0.0;

        // Q blocks
        // For e, e_dot
        auto Q_track = Q.block<num_dof * 2, num_dof * 2>(0, 0);
        // For m, d, k
        auto Q_comp = Q.block<num_dof * 3, num_dof * 3>(num_dof * 2, num_dof * 2);

        // Compliance cost weighting
        mpc::mat<num_inputs, num_inputs> He_R;
        mpc::cvec<num_inputs> he_diag;
        he_diag.segment<num_dof>(0) = external_wrench.array().abs();
        he_diag.segment<num_dof>(num_dof) = external_wrench.array().abs();
        he_diag.segment<num_dof>(num_dof * 2) = external_wrench.array().abs();
        He_R = he_diag.asDiagonal();

        for (int i = 0; i < pred_hor; i++)
        {
            // Tracking Cost
            mpc::cvec<num_dof * 2> track_state = x.row(i).segment<num_dof * 2>(0).transpose();
            cost += track_state.transpose() * Q_track * track_state;

            // Compliance Cost
            mpc::cvec<num_dof * 3> comp_state = x.row(i).segment<num_dof * 3>(num_dof * 2).transpose();
            cost += comp_state.transpose() * He_R.transpose() * Q_comp * He_R * comp_state;

            // Input Variation Cost
            mpc::cvec<num_inputs> u_i = u.row(i).transpose();
            cost += u_i.transpose() * R * u_i;
        }

        // Terminal Cost
        mpc::cvec<num_dof * 2> track_state_T = x.row(pred_hor).segment<num_dof * 2>(0).transpose();
        cost += track_state_T.transpose() * Q_track * track_state_T;
        mpc::cvec<num_dof * 3> comp_state_T = x.row(pred_hor).segment<num_dof * 3>(num_dof * 2).transpose();
        cost += comp_state_T.transpose() * He_R.transpose() * Q_comp * He_R * comp_state_T;

        return cost; 
    });

    // --- 9. Define Constraints (Bounds) ---
    // State bounds
    mpc::cvec<num_dof> m_min, m_max, d_min, d_max, k_min, k_max;
    m_min.setConstant(0.0);
    m_max.setConstant(3.0);
    d_min.setConstant(5.0);
    d_max.setConstant(50.0);
    k_min.setConstant(0.0);
    k_max.setConstant(1000.0);

    mpc::cvec<num_states> x_lower, x_upper;
    x_lower.segment<num_dof>(0).setConstant(-mpc::inf);
    x_upper.segment<num_dof>(0).setConstant(mpc::inf); // e
    x_lower.segment<num_dof>(num_dof).setConstant(-mpc::inf);
    x_upper.segment<num_dof>(num_dof).setConstant(mpc::inf); // e_dot
    x_lower.segment<num_dof>(num_dof * 2) = m_min;
    x_upper.segment<num_dof>(num_dof * 2) = m_max; // m
    x_lower.segment<num_dof>(num_dof * 3) = d_min;
    x_upper.segment<num_dof>(num_dof * 3) = d_max; // d
    x_lower.segment<num_dof>(num_dof * 4) = k_min;
    x_upper.segment<num_dof>(num_dof * 4) = k_max; // k

    if (!controller.setStateBounds(x_lower, x_upper, mpc::HorizonSlice::all()))
    {
        std::cerr << "Failed to set STATE vector bounds!" << std::endl;
        return 1;
    }

    // Input bounds
    mpc::cvec<num_dof> m_dot_min, m_dot_max, d_dot_min, d_dot_max, k_dot_min, k_dot_max;
    m_dot_min.setConstant(-2.0);
    m_dot_max.setConstant(2.0);
    d_dot_min.setConstant(-5.0);
    d_dot_max.setConstant(5.0);
    k_dot_min.setConstant(-10.0);
    k_dot_max.setConstant(10.0);

    mpc::cvec<num_inputs> u_lower, u_upper;
    u_lower.segment<num_dof>(0) = m_dot_min;
    u_upper.segment<num_dof>(0) = m_dot_max; // m_dot
    u_lower.segment<num_dof>(num_dof) = d_dot_min;
    u_upper.segment<num_dof>(num_dof) = d_dot_max; // d_dot
    u_lower.segment<num_dof>(num_dof * 2) = k_dot_min;
    u_upper.segment<num_dof>(num_dof * 2) = k_dot_max; // k_dot

    if (!controller.setInputBounds(u_lower, u_upper, mpc::HorizonSlice(0, ctrl_hor - 1)))
    {
        std::cerr << "Failed to set INPUT vector bounds!" << std::endl;
        return 1;
    }

    // Set last input to zero (terminal constraint)
    mpc::cvec<num_inputs> u_lower_last, u_upper_last;
    u_lower_last.setZero();
    u_upper_last.setZero();
    if (!controller.setInputBounds(u_lower_last, u_upper_last, mpc::HorizonSlice(ctrl_hor - 1, ctrl_hor)))
    {
        std::cerr << "Failed to set terminal INPUT vector bounds!" << std::endl;
        return 1;
    }

    // --- 10. Run Simulation Loop ---

    // Simulation state (absolute)
    mpc::cvec<num_dof> p_sim, p_dot_sim, p_ddot_sim;
    mpc::cvec<num_dof> m_sim, d_sim, k_sim;

    // MPC state (error)
    mpc::cvec<num_states> modelX;

    // Desired trajectory
    mpc::cvec<num_dof> p_des, p_dot_des, p_ddot_des;

    // Set initial "real" state
    p_sim.setZero();
    p_dot_sim.setZero();
    p_ddot_sim.setZero();

    // Set initial impedance
    m_sim = (m_min.array() + m_max.array()) / 2.0;
    d_sim = (d_min.array() + d_max.array()) / 2.0;
    k_sim = (k_min.array() + k_max.array()) / 2.0;

    // Set initial MPC error state
    get_trajectory(0.0, p_des, p_dot_des, p_ddot_des);
    modelX.segment<num_dof>(0) = p_sim - p_des;
    modelX.segment<num_dof>(num_dof) = p_dot_sim - p_dot_des;
    modelX.segment<num_dof>(num_dof * 2) = m_sim;
    modelX.segment<num_dof>(num_dof * 3) = d_sim;
    modelX.segment<num_dof>(num_dof * 4) = k_sim;

    auto r = controller.getLastResult();
    double current_time = 0.0;

    std::cout << "--- 6-DOF Trajectory Tracking Simulation ---" << std::endl;

    for (int i = 0; i < 400; i++)
    {
        current_time = i * ts;
        
        // 2. Get optimal input (derivatives)
        r = controller.optimize(modelX, r.cmd);
        mpc::cvec<num_inputs> u = r.cmd;

        // 3. Simulate the real system for one time step
        get_trajectory(current_time, p_des, p_dot_des, p_ddot_des);
        mpc::cvec<num_dof> e = p_sim - p_des;
        mpc::cvec<num_dof> e_dot = p_dot_sim - p_dot_des;

        mpc::mat<num_dof, num_dof> M = m_sim.asDiagonal();
        mpc::mat<num_dof, num_dof> D = d_sim.asDiagonal();
        mpc::mat<num_dof, num_dof> K = k_sim.asDiagonal();

        mpc::cvec<num_dof> e_ddot = M.inverse() * (external_wrench - K * e - D * e_dot);
        p_ddot_sim = p_ddot_des + e_ddot;

        p_dot_sim += p_ddot_sim * ts;
        p_sim += p_dot_sim * ts;

        m_sim += u.segment<num_dof>(0) * ts;
        d_sim += u.segment<num_dof>(num_dof) * ts;
        k_sim += u.segment<num_dof>(num_dof * 2) * ts;

        // 4. Calculate next error state for the MPC
        get_trajectory(current_time + ts, p_des, p_dot_des, p_ddot_des);
        modelX.segment<num_dof>(0) = p_sim - p_des;
        modelX.segment<num_dof>(num_dof) = p_dot_sim - p_dot_des;
        modelX.segment<num_dof>(num_dof * 2) = m_sim;
        modelX.segment<num_dof>(num_dof * 3) = d_sim;
        modelX.segment<num_dof>(num_dof * 4) = k_sim;

        // 5. Print status
        double err_pos_norm = modelX.segment<num_dof>(0).norm();
        double err_vel_norm = modelX.segment<num_dof>(num_dof).norm();
        auto mass_vec_all = modelX.segment<num_dof>(num_dof * 2);
        auto damp_vec_all = modelX.segment<num_dof>(num_dof * 3);
        auto stiff_vec_all = modelX.segment<num_dof>(num_dof * 4);

        std::cout << i << "    | "
                  << std::fixed << std::setprecision(4)
                  << err_pos_norm << "    | "
                  << err_vel_norm << "    \n "
                  << mass_vec_all.transpose() << "   |\n"
                  << damp_vec_all.transpose() << "   |\n"
                  << stiff_vec_all.transpose() << "\n";

        // check if the results are within bounds
        if ((m_sim.array() < m_min.array()).any() || (m_sim.array() > m_max.array()).any() ||
            (d_sim.array() < d_min.array()).any() || (d_sim.array() > d_max.array()).any() ||
            (k_sim.array() < k_min.array()).any() || (k_sim.array() > k_max.array()).any())
        {
            std::cerr << "Simulation impedance values out of bounds!" << std::endl;
            return 1;
        }
    }

    std::cout << "\n" << controller.getExecutionStats() << std::endl;
    return 0;
}