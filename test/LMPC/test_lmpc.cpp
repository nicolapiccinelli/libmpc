/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

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

    REQUIRE((res.l.segment(0, Tnx).array() == -1).all());
    REQUIRE((res.l.segment(Tnx, Tnu).array() == 1).all());
    REQUIRE(res.l.segment(Tnx + Tnu, Tph * (Tnx + Tnu)).isZero());
    REQUIRE((res.l.segment((Tph + 1) * (Tnx + Tnu), ((Tph + 1) * (Tnu + Tnx)) + ((Tph + 1) * Tny) + (Tph * Tnu) + (Tph + 1)).array() == -mpc::inf).all());

    REQUIRE((res.u.segment(0, Tnx).array() == -1).all());
    REQUIRE((res.u.segment(Tnx, Tnu).array() == 1).all());
    REQUIRE(res.u.segment(Tnx + Tnu, Tph * (Tnx + Tnu)).isZero());
    REQUIRE((res.u.segment((Tph + 1) * (Tnx + Tnu), ((Tph + 1) * (Tnu + Tnx)) + ((Tph + 1) * Tny) + (Tph * Tnu) + (Tph + 1)).array() == mpc::inf).all());
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
    mpc::mat<Tnu, Tph> uminmat, umaxmat;

    xminmat.setConstant(-1);
    xmaxmat.setConstant(1);

    yminmat.setConstant(-2);
    ymaxmat.setConstant(2);

    uminmat.setConstant(-3);
    umaxmat.setConstant(3);

    builder.setConstraints(xminmat, uminmat, yminmat, xmaxmat, umaxmat, ymaxmat);

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
        REQUIRE((res.l.segment(((Tph + 1) * (Tnu + Tnx)) + expected_xu_l.size() + expected_y_l.size(), Tph * Tnu).array() == -mpc::inf).all());
        REQUIRE((res.u.segment(((Tph + 1) * (Tnu + Tnx)) + expected_xu_u.size() + expected_y_u.size(), Tph * Tnu).array() == mpc::inf).all());
    }

    REQUIRE((res.l.tail(Tph).array() == -4).all());
    REQUIRE((res.u.tail(Tph).array() == 4).all());
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
    builder.setExogenuosInput(Bdv, Ddv);

    mpc::cvec<Tnx> x;
    x.setRandom();
    mpc::cvec<Tndu> du;
    du.setRandom();

    REQUIRE(builder.mapToOutput(x, du).isApprox(Cd * x + Ddv * du));
}