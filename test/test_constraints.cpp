#include "basic.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking missing user constraints"), 
    MPC_TEST_TAGS("[constraints][template]"),
    ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
    (1, 1, 1, 1, 1, 0), (5, 1, 1, 1, 1, 0), (5, 3, 1, 1, 1, 0),
    (5, 3, 1, 7, 1, 0), (5, 3, 1, 7, 4, 0), (5, 3, 1, 7, 7, 0))
{
    constexpr int Teq = Tineq;

    mpc::ConFunction<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> conFunc;
    conFunc.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    REQUIRE_FALSE(conFunc.hasEqConstraintFunction());
    REQUIRE_FALSE(conFunc.hasIneqConstraintFunction());
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking model equality constraints"), 
    MPC_TEST_TAGS("[constraints][template]"),
    ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
    (2, 1, 1, 5, 5, 0))
{
    constexpr int Teq = 0;

    mpc::ConFunction<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> conFunc;
    conFunc.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    mpc::Mapping<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> mapping;
    mapping.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    conFunc.setMapping(mapping);
    conFunc.setContinuos(true);
    conFunc.setStateSpaceFunction([](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> &dx,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> u)
        {
            dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
            dx[1] = x[0];
        });

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x0;
    x0.resize(Tnx);
    x0 << 0, 0;
    conFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<MPC_DYNAMIC_TEST_VAR((mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize)))> x;
    x.resize(mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize));
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    mpc::cvec<MPC_DYNAMIC_TEST_VAR((mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::StateEqSize)))> costExpected;
    costExpected.resize(mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::StateEqSize));
    costExpected << 0, -1, -2, -2, -2, -2, -2, -2, -2, -2;

    auto c = conFunc.evaluateEq(x, false);

    REQUIRE(c.value == costExpected);
    REQUIRE(c.grad.isZero());
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking user inequality constraints"), 
    MPC_TEST_TAGS("[constraints][template]"),
    ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
    (2, 1, 1, 5, 5, 1))
{
    constexpr int Teq = 0;

    mpc::ConFunction<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> conFunc;
    conFunc.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    mpc::Mapping<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> mapping;
    mapping.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    conFunc.setMapping(mapping);
    conFunc.setContinuos(true);
    conFunc.setStateSpaceFunction([](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> &dx,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> u) 
    {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
    });

    conFunc.setIneqConstraintFunction([](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tineq)> &eq_con,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnu)> u,
        double e) 
    {
        for (int i = 0; i < Tineq; i++)
        {
            eq_con[i] = x(0,0);
        }
    });

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x0;
    x0.resize(Tnx);
    x0 << 10, 0;
    conFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<MPC_DYNAMIC_TEST_VAR((mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize)))> x;
    x.resize(mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize));
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i + 1;
    }

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tineq)> costExpected;
    costExpected.resize(Tineq);
    costExpected << x0[0];

    auto c = conFunc.evaluateUserIneq(x, false);

    REQUIRE(c.value == costExpected);
    REQUIRE(c.grad.isZero());
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking user equality constraints"), 
    MPC_TEST_TAGS("[constraints][template]"),
    ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
    (2, 1, 1, 5, 5, 0))
{
    constexpr int Teq = 1;

    mpc::ConFunction<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> conFunc;
    conFunc.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    mpc::Mapping<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> mapping;
    mapping.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    conFunc.setMapping(mapping);
    conFunc.setContinuos(true);
    conFunc.setStateSpaceFunction([](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> &dx,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> u) 
    {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
    });

    conFunc.setEqConstraintFunction([Teq](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Teq)> &eq_con,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnu)> u) 
    {
        for (int i = 0; i < Teq; i++)
        {
            eq_con[i] = x(0, 0);
        }
    });

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x0;
    x0.resize(Tnx);
    x0 << 10, 0;
    conFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<MPC_DYNAMIC_TEST_VAR((mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize)))> x;
    x.resize(mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize));
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i + 1;
    }

    mpc::cvec<Teq> costExpected;
    costExpected.resize(Teq);
    costExpected << x0[0];

    auto c = conFunc.evaluateUserEq(x, false);

    REQUIRE(c.value == costExpected);
    REQUIRE(c.grad.isZero());
}