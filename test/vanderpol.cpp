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

    mpc::NLMPC<num_states, num_inputs, num_output, pred_hor, ctrl_hor, ineq_c, eq_c> optsolver(useHardConst, true);
    optsolver.setVerbosity(true, mpc::Logger::INFO);

    auto stateEq = [](mpc::cvec<num_states> x, mpc::cvec<num_inputs> u) {
        mpc::cvec<num_states> dx;
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
        return dx;
    };

    optsolver.setStateSpaceFunction(stateEq);
    optsolver.setObjectiveFunction([](mpc::mat<pred_hor + 1, num_states> x, mpc::mat<pred_hor + 1, num_inputs> u, double e) {
        return x.array().square().sum() + u.array().square().sum();
    });

    optsolver.setIneqConFunction([](mpc::mat<pred_hor + 1, num_states> x, mpc::mat<pred_hor + 1, num_inputs> u, double e) {
        mpc::cvec<ineq_c> in_con;
        for (size_t i = 0; i < ineq_c; i++) {
            in_con[i] = u(i, 0) - 0.5;
        }

        return in_con;
    });

    optsolver.setSampleTime(ts);

    mpc::cvec<num_states> modelX;
    modelX[0] = 0;
    modelX[1] = 1.0;

    mpc::Result<num_inputs> r;
    r.cmd[0] = 0.0;

    for (;;) {
        r = optsolver.step(modelX, r.cmd);
        modelX += stateEq(modelX, r.cmd) * ts;
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