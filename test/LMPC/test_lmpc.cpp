/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEST_CASE(
    MPC_TEST_NAME("State box constraints"),
    MPC_TEST_TAGS("[linear]"))
{
    constexpr int Tnx = 2;
    constexpr int Tny = 2;
    constexpr int Tnu = 1;
    constexpr int Tndu = 0;
    constexpr int Tph = 15;
    constexpr int Tch = 15;

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

    optsolver.setLoggerLevel(mpc::Logger::log_level::DEEP);

    mpc::mat<Tnx, Tnx> A, Ad;
    A << 0, 1, 0, 2;
    mpc::mat<Tnx, Tnu> B, Bd;
    B << 0, 1;

    mpc::discretization<Tnx, Tnu>(A, B, 0.01, Ad, Bd);

    mpc::mat<Tny, Tnx> C;
    C.setIdentity();

    optsolver.setStateSpaceModel(Ad, Bd, C);

    mpc::cvec<Tnu> InputW, DeltaInputW;
    mpc::cvec<Tny> OutputW;

    OutputW << 0, 0;
    InputW << 0;
    DeltaInputW << 0;

    REQUIRE(optsolver.setObjectiveWeights(OutputW, InputW, DeltaInputW, mpc::HorizonSlice::all()));
    REQUIRE(optsolver.setReferences(mpc::mat<Tny, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero()));

    mpc::mat<Tnx, Tph> xminmat, xmaxmat;
    mpc::mat<Tny, Tph> yminmat, ymaxmat;
    mpc::mat<Tnu, Tch> uminmat, umaxmat;

    xminmat.setConstant(-mpc::inf);
    xmaxmat.setConstant(mpc::inf);
    xminmat.col(Tph - 1) << 0.0, 0.0;
    xmaxmat.col(Tph - 1) << 0.0, 0.0;

    yminmat.setConstant(-mpc::inf);
    ymaxmat.setConstant(mpc::inf);

    uminmat.setConstant(-mpc::inf);
    umaxmat.setConstant(mpc::inf);

    optsolver.setStateBounds(xminmat, xmaxmat);
    optsolver.setInputBounds(uminmat, umaxmat);
    optsolver.setOutputBounds(yminmat, ymaxmat);

    mpc::cvec<Tnx> x;
    x << 2.0, 0;
    mpc::cvec<Tnu> u;
    u << 0;

    mpc::LParameters params;
    params.maximum_iteration = 4000;
    params.verbose = true;
    optsolver.setOptimizerParameters(params);

    auto res = optsolver.optimize(x, u);
    auto seq = optsolver.getOptimalSequence();

    for (size_t i = 0; i < Tph; i++)
    {
        std::cout << seq.state.row(i) << std::endl;
    }

    optsolver.setLoggerLevel(mpc::Logger::log_level::NONE);
}

TEST_CASE(
    MPC_TEST_NAME("Scalar constraints"),
    MPC_TEST_TAGS("[linear]"))
{
    constexpr int Tnx = 2;
    constexpr int Tny = 2;
    constexpr int Tnu = 1;
    constexpr int Tndu = 0;
    constexpr int Tph = 5;
    constexpr int Tch = 5;

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

    mpc::mat<Tnx, Tnx> A, Ad;
    A << 0, 1, 0, 2;
    mpc::mat<Tnx, Tnu> B, Bd;
    B << 0, 1;

    mpc::discretization<Tnx, Tnu>(A, B, 0.001, Ad, Bd);

    mpc::mat<Tny, Tnx> C;
    C.setIdentity();

    optsolver.setStateSpaceModel(Ad, Bd, C);

    mpc::cvec<Tnu> InputW, DeltaInputW;
    mpc::cvec<Tny> OutputW;

    OutputW << 1, 0;
    InputW << 0.1;
    DeltaInputW << 0;

    REQUIRE(optsolver.setObjectiveWeights(OutputW, InputW, DeltaInputW, {-1, -1}));

    mpc::cvec<Tnx> sX;
    sX.setOnes();
    mpc::cvec<Tnu> sU;
    sU.setOnes();
    double maxS, minS;
    maxS = 0.1;
    minS = -0.5;

    REQUIRE(optsolver.setScalarConstraint(minS, maxS, sX, sU, {-1, -1}));
    REQUIRE(optsolver.setReferences(mpc::mat<Tny, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero()));

    mpc::cvec<Tnx> x;
    x << 10.0, 0;
    mpc::cvec<Tnu> u;
    u << 0;

    mpc::LParameters params;
    params.maximum_iteration = 4000;
    optsolver.setOptimizerParameters(params);

    auto res = optsolver.optimize(x, u);
    auto seq = optsolver.getOptimalSequence();

    for (size_t i = 0; i < Tph; i++)
    {
        double scalar_res = sU.dot(seq.input.row(i)) + sX.dot(seq.state.row(i));
        REQUIRE(scalar_res <= maxS + 1e-2);
        REQUIRE(scalar_res >= minS - 1e-3);
    }
}

TEST_CASE(
    MPC_TEST_NAME("Linear default constraints"),
    MPC_TEST_TAGS("[linear]"))
{
    constexpr int Tnx = 3;
    constexpr int Tny = 4;
    constexpr int Tnu = 5;
    constexpr int Tndu = 6;
    constexpr int Tph = 5;
    constexpr int Tch = 5;

    mpc::ProblemBuilder<mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(Tndu), TVAR(Tny), TVAR(Tph), TVAR(Tch), 0, 0)> builder;
    builder.initialize(Tnx, Tnu, Tndu, Tny, Tph, Tch);

    mpc::cvec<Tnx> x0;
    x0.fill(1);
    mpc::cvec<Tnu> u0;
    u0.fill(-1);

    mpc::mat<Tny, Tph> yRef;
    mpc::mat<Tnu, Tph> uRef;
    mpc::mat<Tnu, Tph> deltaURef;
    mpc::mat<Tndu, Tph> uMeas;

    auto &res = builder.get(x0, u0, yRef, uRef, deltaURef, uMeas);

    REQUIRE((-1 == res.l.segment(0, Tnx).array()).all());
    REQUIRE((1 == res.l.segment(Tnx, Tnu).array()).all());
    REQUIRE(res.l.segment(Tnx + Tnu, Tph * (Tnx + Tnu)).isZero());
    REQUIRE((-mpc::inf == res.l.segment((Tph + 1) * (Tnx + Tnu), ((Tph + 1) * (Tnu + Tnx)) + ((Tph + 1) * Tny) + (Tph * Tnu) + (Tph + 1)).array()).all());

    REQUIRE((-1 == res.u.segment(0, Tnx).array()).all());
    REQUIRE((1 == res.u.segment(Tnx, Tnu).array()).all());
    REQUIRE(res.u.segment(Tnx + Tnu, Tph * (Tnx + Tnu)).isZero());
    REQUIRE((mpc::inf == res.u.segment((Tph + 1) * (Tnx + Tnu), ((Tph + 1) * (Tnu + Tnx)) + ((Tph + 1) * Tny) + (Tph * Tnu) + (Tph + 1)).array()).all());
}

TEST_CASE(
    MPC_TEST_NAME("Linear constraints")
        MPC_TEST_TAGS("[linear]"))
{
    constexpr int Tnx = 2;
    constexpr int Tny = 3;
    constexpr int Tnu = 4;
    constexpr int Tndu = 0;
    constexpr int Tph = 3;
    constexpr int Tch = 3;

    mpc::ProblemBuilder<mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(Tndu), TVAR(Tny), TVAR(Tph), TVAR(Tch), 0, 0)> builder;
    builder.initialize(Tnx, Tnu, Tndu, Tny, Tph, Tch);

    mpc::mat<Tnx, Tph> xminmat, xmaxmat;
    mpc::mat<Tny, Tph> yminmat, ymaxmat;
    mpc::mat<Tnu, Tch> uminmat, umaxmat;

    xminmat.setConstant(-1);
    xmaxmat.setConstant(1);

    yminmat.setConstant(-2);
    ymaxmat.setConstant(2);

    uminmat.setConstant(-3);
    umaxmat.setConstant(3);

    builder.setStateBounds(xminmat, xmaxmat);
    builder.setInputBounds(uminmat, umaxmat);
    builder.setOutputBounds(yminmat, ymaxmat);

    mpc::cvec<Tnx> x0;
    x0.fill(42.0);
    mpc::cvec<Tnu> u0;
    u0.fill(-42.0);

    mpc::cvec<Tph> smin, smax;
    smin.setConstant(-4);
    smax.setConstant(4);

    builder.setScalarConstraint(smin, smax, x0, u0);

    mpc::mat<Tny, Tph> yRef;
    yRef.setZero();
    mpc::mat<Tnu, Tph> uRef;
    uRef.setZero();
    mpc::mat<Tnu, Tph> deltaURef;
    deltaURef.setZero();
    mpc::mat<Tndu, Tph> uMeas;
    uMeas.setZero();

    auto &res = builder.get(x0, u0, yRef, uRef, deltaURef, uMeas);

    mpc::cvec<(Tph + 1) * (Tnu + Tnx)> expected_xu_l, expected_xu_u;
    for (size_t i = 0; i < Tph + 1; i++)
    {
        mpc::cvec<Tnx> t;
        mpc::cvec<Tnu> tt;

        t.setConstant(-1);
        tt.setConstant(-3);
        expected_xu_l.segment(i * (Tnu + Tnx), Tnu + Tnx) << t, tt;

        t.setConstant(1);
        tt.setConstant(3);
        expected_xu_u.segment(i * (Tnu + Tnx), Tnu + Tnx) << t, tt;
    }

    REQUIRE(res.l.segment((Tph + 1) * (Tnu + Tnx), expected_xu_l.size()).isApprox(expected_xu_l));
    REQUIRE(res.u.segment((Tph + 1) * (Tnu + Tnx), expected_xu_u.size()).isApprox(expected_xu_u));

    mpc::cvec<(Tph + 1) * Tny> expected_y_l, expected_y_u;
    for (size_t i = 0; i < Tph + 1; i++)
    {
        expected_y_l.segment(i * Tny, Tny).setConstant(-2);
        expected_y_u.segment(i * Tny, Tny).setConstant(2);
    }

    REQUIRE(res.l.segment(((Tph + 1) * (Tnu + Tnx)) + expected_xu_l.size(), expected_y_l.size()).isApprox(expected_y_l));
    REQUIRE(res.u.segment(((Tph + 1) * (Tnu + Tnx)) + expected_xu_u.size(), expected_y_u.size()).isApprox(expected_y_u));

    if (Tch >= Tph)
    {
        REQUIRE((-mpc::inf == res.l.segment(((Tph + 1) * (Tnu + Tnx)) + expected_xu_l.size() + expected_y_l.size(), Tph * Tnu).array()).all());
        REQUIRE((mpc::inf == res.u.segment(((Tph + 1) * (Tnu + Tnx)) + expected_xu_u.size() + expected_y_u.size(), Tph * Tnu).array()).all());
    }

    REQUIRE((-4 == res.l.tail(Tph).array()).all());
    REQUIRE((4 == res.u.tail(Tph).array()).all());
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