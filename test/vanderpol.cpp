#include <mpc/nlmpc.hpp>

int main(void)
{
    std::cout << "NLMPC test" << std::endl;

    constexpr int num_states = 2;
    constexpr int num_output = 2;
    constexpr int num_inputs = 1;
    constexpr int pred_hor = 10;
    constexpr int ctrl_hor = 5;
    constexpr int ineq_c = pred_hor + 1;
    constexpr int eq_c = 0;

    double ts = 0.1;

    bool useHardConst = true;

    mpc::NLMPC<num_states, num_inputs, num_output, pred_hor, ctrl_hor, ineq_c, eq_c> optsolver;

    optsolver.initialize(useHardConst);
    optsolver.setLoggerLevel(mpc::Logger::NONE);
    optsolver.setSampleTime(ts);

    auto stateEq = [](mpc::cvec<num_states>& dx, mpc::cvec<num_states> x, mpc::cvec<num_inputs> u) {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
    };

    optsolver.setStateSpaceFunction(stateEq);
    optsolver.setObjectiveFunction([](mpc::mat<pred_hor + 1, num_states> x, mpc::mat<pred_hor + 1, num_inputs> u, double e) {
        return x.array().square().sum() + u.array().square().sum();
    });

    optsolver.setIneqConFunction([](mpc::cvec<ineq_c>& in_con, mpc::mat<pred_hor + 1, num_states> x, mpc::mat<pred_hor + 1, num_inputs> u, double e) {
        for (size_t i = 0; i < ineq_c; i++) {
            in_con[i] = u(i, 0) - 0.5;
        }
    });


    mpc::cvec<num_states> modelX, modeldX;
    modelX[0] = 0;
    modelX[1] = 1.0;

    mpc::Result<num_inputs> r;
    r.cmd[0] = 0.0;

    for (;;) {
        r = optsolver.step(modelX, r.cmd);
        stateEq(modeldX, modelX, r.cmd);
        modelX += modeldX * ts;
        std::cout << "Command: \n"
                  << r.cmd << std::endl;
        std::cout << "States: \n"
                  << modelX << std::endl;
        if (std::fabs(modelX[0]) <= 1e-2 && std::fabs(modelX[1]) <= 1e-1) {
            break;
        }
    }

    std::cout << "NLMPC test done" << std::endl;

    return 0;
}