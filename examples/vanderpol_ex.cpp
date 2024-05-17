/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include <mpc/NLMPC.hpp>

int main()
{
    constexpr int num_states = 2;
    constexpr int num_output = 2;
    constexpr int num_inputs = 1;
    constexpr int pred_hor = 10;
    constexpr int ctrl_hor = 5;
    constexpr int ineq_c = pred_hor + 1;
    constexpr int eq_c = 0;

    double ts = 0.1;

    mpc::NLMPC<
        num_states, num_inputs, num_output,
        pred_hor, ctrl_hor,
        ineq_c, eq_c>
        controller;

    controller.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    controller.setDiscretizationSamplingTime(ts);

    auto stateEq = [&](
                       mpc::cvec<num_states> &dx,
                       const mpc::cvec<num_states> &x,
                       const mpc::cvec<num_inputs> &u)
    {
        dx(0) = ((1.0 - (x(1) * x(1))) * x(0)) - x(1) + u(0);
        dx(1) = x(0);
    };

    controller.setStateSpaceFunction([&](
                                        mpc::cvec<num_states> &dx,
                                        const mpc::cvec<num_states> &x,
                                        const mpc::cvec<num_inputs> &u,
                                        const unsigned int &)
                                    { stateEq(dx, x, u); });

    controller.setObjectiveFunction([&](
                                       const mpc::mat<pred_hor + 1, num_states> &x,
                                       const mpc::mat<pred_hor + 1, num_output> &,
                                       const mpc::mat<pred_hor + 1, num_inputs> &u,
                                       double)
                                   { return x.array().square().sum() + u.array().square().sum(); });

    controller.setIneqConFunction([&](
                                     mpc::cvec<ineq_c> &in_con,
                                     const mpc::mat<pred_hor + 1, num_states> &,
                                     const mpc::mat<pred_hor + 1, num_output> &,
                                     const mpc::mat<pred_hor + 1, num_inputs> &u,
                                     const double &)
                                 {
        for (int i = 0; i < ineq_c; i++) {
            in_con(i) = u(i, 0) - 0.5;
        } });

    mpc::cvec<num_states> modelX, modeldX;
    modelX.resize(num_states);
    modeldX.resize(num_states);

    modelX(0) = 0;
    modelX(1) = 1.0;

    auto r = controller.getLastResult();

    for (;;)
    {
        r = controller.optimize(modelX, r.cmd);
        stateEq(modeldX, modelX, r.cmd);
        modelX += modeldX * ts;
        if (std::fabs(modelX[0]) <= 1e-2 && std::fabs(modelX[1]) <= 1e-1)
        {
            break;
        }
    }

    std::cout << controller.getExecutionStats();

    return 0;
}