/*
 *   Copyright (c) 2023-2025 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEST_CASE(
    MPC_TEST_NAME("Linear multiple instances"),
    MPC_TEST_TAGS("[linear]"))
{
    mpc::Logger::instance().setLevel(mpc::Logger::LogLevel::DEEP);

    // MPC LATERAL CONTROLLER
    constexpr int num_states_1 = 8;
    constexpr int num_output_1 = 2;
    constexpr int num_inputs_1 = 2;
    constexpr int num_dinputs_1 = 5;

    constexpr int num_states_2 = 4;
    constexpr int num_output_2 = 1;
    constexpr int num_inputs_2 = 1;
    constexpr int num_dinputs_2 = 3;

    constexpr int pred_hor = 10;
    constexpr int ctrl_hor = 3;

    // MPC LATERAL CONTROLLER
#ifdef MPC_DYNAMIC
    mpc::LMPC<> latController(
        num_states_1, num_inputs_1, num_dinputs_1, num_output_1,
        pred_hor, ctrl_hor);
#else
    mpc::LMPC<
        TVAR(num_states_1), TVAR(num_inputs_1), TVAR(num_dinputs_1), TVAR(num_output_1),
        TVAR(pred_hor), TVAR(ctrl_hor)>
        latController;
#endif

    mpc::mat<num_states_1, num_states_1> Ad;
    Ad.setIdentity();
    mpc::mat<num_states_1, num_inputs_1> Bd;
    Bd.setZero();
    mpc::mat<num_output_1, num_states_1> Cd;
    Cd.setZero();

    latController.setStateSpaceModel(Ad, Bd, Cd);
    latController.getLastResult();

    // MPC LONGITUDINAL CONTROLLER
#ifdef MPC_DYNAMIC
    mpc::LMPC<> longController(
        num_states_1, num_inputs_1, num_dinputs_1, num_output_1,
        pred_hor, ctrl_hor);
#else
    mpc::LMPC<
        TVAR(num_states_1), TVAR(num_inputs_1), TVAR(num_dinputs_1), TVAR(num_output_1),
        TVAR(pred_hor), TVAR(ctrl_hor)>
        longController;
#endif

    longController.setStateSpaceModel(Ad, Bd, Cd);
    longController.getLastResult();

    // MPC VERTICAL CONTROLLER
#ifdef MPC_DYNAMIC
    mpc::LMPC<> vertController(
        num_states_2, num_inputs_2, num_dinputs_2, num_output_2,
        pred_hor, ctrl_hor);
#else
    mpc::LMPC<
        TVAR(num_states_2), TVAR(num_inputs_2), TVAR(num_dinputs_2), TVAR(num_output_2),
        TVAR(pred_hor), TVAR(ctrl_hor)>
        vertController;
#endif

    mpc::mat<num_states_2, num_states_2> Ad_2;
    Ad_2.setIdentity();
    mpc::mat<num_states_2, num_inputs_2> Bd_2;
    Bd_2.setZero();
    mpc::mat<num_output_2, num_states_2> Cd_2;
    Cd_2.setIdentity();

    vertController.setStateSpaceModel(Ad_2, Bd_2, Cd_2);
    vertController.getLastResult();
}

TEST_CASE(
    MPC_TEST_NAME("LMPC interface test"),
    MPC_TEST_TAGS("[linear]"))
{
    constexpr int Tnx = 12;
    constexpr int Tny = 12;
    constexpr int Tnu = 4;
    constexpr int Tndu = 4;
    constexpr int Tph = 10;
    constexpr int Tch = 10;

#ifdef MPC_DYNAMIC
    mpc::LMPC<> optsolver(
        Tnx, Tnu, Tndu, Tny,
        Tph, Tch);
#else
    mpc::LMPC<
        TVAR(Tnx), TVAR(Tnu), TVAR(Tndu), TVAR(Tny),
        TVAR(Tph), TVAR(Tch)>
        optsolver;
#endif

    optsolver.setLoggerLevel(mpc::Logger::LogLevel::NONE);

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

    optsolver.setStateSpaceModel(Ad, Bd, Cd);

    optsolver.setDisturbances(
        mpc::mat<Tnx, Tndu>::Zero(),
        mpc::mat<Tny, Tndu>::Zero());

    mpc::mat<Tnu, Tph> InputWMat, DeltaInputWMat;
    mpc::mat<Tny, Tph> OutputWMat;

    REQUIRE(optsolver.setObjectiveWeights(OutputWMat, InputWMat, DeltaInputWMat));

    mpc::cvec<Tnu> InputW, DeltaInputW;
    mpc::cvec<Tny> OutputW;

    OutputW << 0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5;
    InputW << 0.1, 0.1, 0.1, 0.1;
    DeltaInputW << 0, 0, 0, 0;

    REQUIRE(optsolver.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, Tph}));

    mpc::mat<Tnx, Tph> xminmat, xmaxmat;
    mpc::mat<Tny, Tph> yminmat, ymaxmat;
    mpc::mat<Tnu, Tph> uminmat, umaxmat;

    xminmat.setZero();
    xmaxmat.setZero();
    yminmat.setZero();
    ymaxmat.setZero();
    uminmat.setZero();
    umaxmat.setZero();

    REQUIRE(optsolver.setStateBounds(xminmat, xmaxmat));
    REQUIRE(optsolver.setInputBounds(uminmat, umaxmat));
    REQUIRE(optsolver.setOutputBounds(yminmat, ymaxmat));

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

    REQUIRE(optsolver.setStateBounds(xmin, xmax, {0, Tph}));
    REQUIRE(optsolver.setInputBounds(umin, umax, {0, Tph}));
    REQUIRE(optsolver.setOutputBounds(ymin, ymax, {0, Tph}));

    REQUIRE(optsolver.setStateBounds(xmin, xmax, {0, 1}));
    REQUIRE(optsolver.setInputBounds(umin, umax, {0, 1}));
    REQUIRE(optsolver.setOutputBounds(ymin, ymax, {0, 1}));
    
    REQUIRE(optsolver.setScalarConstraint(-mpc::inf, mpc::inf, mpc::cvec<Tnx>::Ones(), mpc::cvec<Tnu>::Ones(), {-1, -1}));
    REQUIRE(optsolver.setScalarConstraint(0, -mpc::inf, mpc::inf, mpc::cvec<Tnx>::Ones(), mpc::cvec<Tnu>::Ones()));

    REQUIRE(optsolver.setReferences(mpc::mat<Tny, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero()));

    mpc::cvec<Tny> yRef;
    yRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    REQUIRE(optsolver.setReferences(yRef, mpc::cvec<Tnu>::Zero(), mpc::cvec<Tnu>::Zero(), {0, Tph}));

    mpc::LParameters params;
    params.maximum_iteration = 250;
    optsolver.setOptimizerParameters(params);

    REQUIRE(optsolver.setExogenousInputs(mpc::mat<Tndu, Tph>::Zero()));
    REQUIRE(optsolver.setExogenousInputs(mpc::cvec<Tndu>::Zero(), {0, Tph}));

    auto res = optsolver.optimize(mpc::cvec<Tnx>::Zero(), mpc::cvec<Tnu>::Zero());
    auto seq = optsolver.getOptimalSequence();
    (void)seq;

    mpc::cvec<4> testRes;
    testRes << -0.9916, 1.74839, -0.9916, 1.74839;

    std::cout << "Expected result: " << testRes << std::endl;
    std::cout << "Obtained result: " << res.cmd << std::endl;

    REQUIRE(res.cmd.isApprox(testRes, 1e-4));
}

TEST_CASE(
    MPC_TEST_NAME("Linear output mapping"),
    MPC_TEST_TAGS("[linear]"))
{
    constexpr int Tnx = 3;
    constexpr int Tny = 6;
    constexpr int Tnu = 0;
    constexpr int Tndu = 7;
    constexpr int Tph = 1;
    constexpr int Tch = 1;

    mpc::ProblemBuilder<mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(Tndu), TVAR(Tny), TVAR(Tph), TVAR(Tch), 0, 0)> builder;
    builder.initialize(Tnx, Tnu, Tndu, Tny, Tph, Tch);

    mpc::mat<Tnx, Tnx> Ad;
    Ad.setZero();

    mpc::mat<Tnx, Tnu> Bd;
    Bd.setZero();

    mpc::mat<Tny, Tnx> Cd;
    Cd.setRandom();

    mpc::mat<Tny, Tnu> Dd;
    Dd.setZero();

    mpc::mat<Tnx, Tndu> Bdv;
    Bdv.setZero();

    mpc::mat<Tny, Tndu> Ddv;
    Ddv.setRandom();

    builder.setStateModel(Ad, Bd, Cd);
    builder.setExogenousInput(Bdv, Ddv);

    mpc::cvec<Tnx> x;
    x.setRandom();
    mpc::cvec<Tndu> du;
    du.setRandom();

    REQUIRE(builder.mapToOutput(x, du).isApprox(Cd * x + Ddv * du));
}