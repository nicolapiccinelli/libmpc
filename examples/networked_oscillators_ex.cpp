#include <iostream>
#include <Eigen/Dense>
#include <mpc/NLMPC.hpp>

constexpr int N = 4;  // Number of oscillators
constexpr int num_states = 2 * N;
constexpr int num_output = 2 * N;
constexpr int num_inputs = N;
constexpr int pred_hor = 20;
constexpr int ctrl_hor = 10;
constexpr int ineq_c = pred_hor + 1;
constexpr int eq_c = 0;
double ts = 0.1;
double mu = 1.0;
double k = 0.1;

void oscillatorNetworkDynamics(mpc::cvec<num_states> &dx, const mpc::cvec<num_states> &x, const mpc::cvec<num_inputs> &u)
{
    for (int i = 0; i < N; ++i) {
        dx(2 * i) = x(2 * i + 1);
        dx(2 * i + 1) = mu * (1 - x(2 * i) * x(2 * i)) * x(2 * i + 1) - x(2 * i) + u(i);

        for (int j = 0; j < N; ++j) {
            if (i != j) {
                dx(2 * i + 1) += k * (x(2 * j) - x(2 * i));
            }
        }
    }
}

int main() {
    mpc::NLMPC<
        num_states, num_inputs, num_output,
        pred_hor, ctrl_hor,
        ineq_c, eq_c>
        controller;

    controller.setLoggerLevel(mpc::Logger::log_level::NONE);
    controller.setDiscretizationSamplingTime(ts);

    controller.setStateSpaceFunction([](
                                        mpc::cvec<num_states> &dx,
                                        const mpc::cvec<num_states> &x,
                                        const mpc::cvec<num_inputs> &u,
                                        const unsigned int &)
                                    { oscillatorNetworkDynamics(dx, x, u); });

    controller.setObjectiveFunction([](
                                       const mpc::mat<pred_hor + 1, num_states> &x,
                                       const mpc::mat<pred_hor + 1, num_output> &,
                                       const mpc::mat<pred_hor + 1, num_inputs> &u,
                                       double)
                                   { return x.array().square().sum() + u.array().square().sum(); });

    controller.setIneqConFunction([](
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
    modelX.setZero();
    modelX(0) = 1.0;  // Initial condition for one of the oscillators

    auto r = controller.getLastResult();

    for (int step = 0; step < 10; ++step)
    {
        r = controller.optimize(modelX, r.cmd);
        oscillatorNetworkDynamics(modeldX, modelX, r.cmd);
        modelX += modeldX * ts;
        if (modelX.array().abs().maxCoeff() < 1e-2)
        {
            break;
        }
    }

    std::cout << controller.getExecutionStats() << std::endl;

    return 0;
}
