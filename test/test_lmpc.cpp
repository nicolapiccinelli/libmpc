#include "basic.hpp"
#include <catch2/catch.hpp>

TEST_CASE(
    MPC_TEST_NAME("Linear example"),
    MPC_TEST_TAGS("[linear]"))
{
    constexpr int num_states = 12;
    constexpr int num_output = 12;
    constexpr int num_inputs = 4;
    constexpr int num_dinputs = 4;
    constexpr int pred_hor = 10;
    constexpr int ctrl_hor = 10;

#ifdef MPC_DYNAMIC
    mpc::LMPC<> optsolver(
        num_states, num_inputs, num_dinputs, num_output,
        pred_hor, ctrl_hor);
#else
    mpc::LMPC<
        TVAR(num_states), TVAR(num_inputs), TVAR(num_dinputs), TVAR(num_output),
        TVAR(pred_hor), TVAR(ctrl_hor)> optsolver;
#endif

    optsolver.setLoggerLevel(mpc::Logger::log_level::NONE);

    mpc::mat<num_states, num_states> Ad;
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

    mpc::mat<num_states, num_inputs> Bd;
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

    mpc::mat<num_output, num_states> Cd;
    Cd.setIdentity();

    mpc::mat<num_output, num_inputs> Dd;
    Dd.setZero();

    optsolver.setStateSpaceModel(Ad, Bd, Cd);

    optsolver.setDisturbances(
        mpc::mat<num_states, num_dinputs>::Zero(),
        mpc::mat<num_output, num_dinputs>::Zero());

    mpc::cvec<num_inputs> InputW, DeltaInputW;
    mpc::cvec<num_output> OutputW;

    OutputW << 0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5;
    InputW << 0.1, 0.1, 0.1, 0.1;
    DeltaInputW << 0, 0, 0, 0;

    optsolver.setObjectiveWeights(OutputW, InputW, DeltaInputW);

    mpc::cvec<num_states> xmin, xmax;
    xmin << -M_PI / 6, -M_PI / 6, -mpc::inf, -mpc::inf, -mpc::inf, -1,
        -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf;

    xmax << M_PI / 6, M_PI / 6, mpc::inf, mpc::inf, mpc::inf, mpc::inf,
        mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf;

    mpc::cvec<num_output> ymin, ymax;
    ymin.setOnes();
    ymin *= -mpc::inf;
    ymax.setOnes();
    ymax *= mpc::inf;

    mpc::cvec<num_inputs> umin, umax;
    double u0 = 10.5916;
    umin << 9.6, 9.6, 9.6, 9.6;
    umin.array() -= u0;
    umax << 13, 13, 13, 13;
    umax.array() -= u0;

    optsolver.setConstraints(xmin, umin, ymin, xmax, umax, ymax);

    mpc::cvec<num_output> yRef;
    yRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    optsolver.setReferences(yRef, mpc::cvec<num_inputs>::Zero(), mpc::cvec<num_inputs>::Zero());
    
    mpc::LParameters params;
    params.maximum_iteration = 250;
    optsolver.setOptimizerParameters(params);

    auto res = optsolver.step(mpc::cvec<num_states>::Zero(), mpc::cvec<num_inputs>::Zero());

    mpc::cvec<4> testRes;
    testRes << -0.9916, 1.74839, -0.9916, 1.74839;

    REQUIRE(res.cmd.isApprox(testRes, 1e-4));
}