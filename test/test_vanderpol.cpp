#include "basic.hpp"
#include <catch2/catch.hpp>

int VanderPol()
{
    constexpr int num_states = 2;
    constexpr int num_output = 2;
    constexpr int num_inputs = 1;
    constexpr int pred_hor = 10;
    constexpr int ctrl_hor = 5;
    constexpr int ineq_c = pred_hor + 1;
    constexpr int eq_c = 0;

    double ts = 0.1;

    bool useHardConst = true;

    mpc::NLMPC<
            num_states,
            num_inputs,
            num_output,
            pred_hor,
            ctrl_hor,
            ineq_c,
            eq_c>
            optsolver;

    optsolver.initialize(useHardConst);
    optsolver.setLoggerLevel(mpc::Logger::log_level::DEEP);
    optsolver.setSampleTime(ts);

    auto stateEq = [](
            mpc::cvec<num_states>& dx,
            mpc::cvec<num_states> x,
            mpc::cvec<num_inputs> u)
    {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
    };
    optsolver.setStateSpaceFunction(stateEq);

    auto objEq = [](
            mpc::mat<pred_hor + 1, num_states> x,
            mpc::mat<pred_hor + 1, num_inputs> u,
            double e)
    {
        return x.array().square().sum() + u.array().square().sum();
    };
    optsolver.setObjectiveFunction(objEq);

    auto conEq = [ineq_c](
            mpc::cvec<ineq_c>& in_con,
            mpc::mat<pred_hor + 1, num_states> x,
            mpc::mat<pred_hor + 1, num_inputs> u,
            double e)
    {
        for (size_t i = 0; i < ineq_c; i++){
            in_con[i] = u(i, 0) - 0.5;
        }
    };
    optsolver.setIneqConFunction(conEq);


    mpc::cvec<num_states> modelX, modeldX;
    modelX[0] = 0;
    modelX[1] = 1.0;

    mpc::Result<num_inputs> r;
    r.cmd[0] = 0.0;

    for (;;) {
        r = optsolver.step(modelX, r.cmd);
        stateEq(modeldX, modelX, r.cmd);
        modelX += modeldX * ts;
        if (std::fabs(modelX[0]) <= 1e-2 && std::fabs(modelX[1]) <= 1e-1) {
            break;
        }
    }

    return 0;
}

TEST_CASE("Vanderpol example", "[vanderpol]")
{
    REQUIRE(VanderPol() == 0);
}