/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking missing user constraints"),
    MPC_TEST_TAGS("[constraints][template]"),
    ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
    (1, 1, 1, 1, 1, 0), (5, 1, 1, 1, 1, 0), (5, 3, 1, 1, 1, 0),
    (5, 3, 1, 7, 1, 0), (5, 3, 1, 7, 4, 0), (5, 3, 1, 7, 7, 0))
{
    constexpr int Teq = Tineq;

    mpc::Constraints<mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(0), TVAR(Tny), TVAR(Tph), TVAR(Tch), TVAR(Tineq), TVAR(Teq))> conFunc;
    conFunc.initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    REQUIRE_FALSE(conFunc.hasEqConstraints());
    REQUIRE_FALSE(conFunc.hasIneqConstraints());
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking model equality constraints"),
    MPC_TEST_TAGS("[constraints][template]"),
    ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
    (2, 1, 1, 5, 5, 0))
{
    constexpr int Teq = 0;
    static constexpr auto sizer = mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(0), TVAR(Tny), TVAR(Tph), TVAR(Tch), TVAR(Tineq), TVAR(Teq));

    std::shared_ptr<mpc::Constraints<sizer>> conFunc;
    conFunc = std::make_shared<mpc::Constraints<sizer>>();
    conFunc->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    std::shared_ptr<mpc::Mapping<sizer>> mapping;
    mapping = std::make_shared<mpc::Mapping<sizer>>();
    mapping->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    std::shared_ptr<mpc::Model<sizer>> model;
    model = std::make_shared<mpc::Model<sizer>>();
    model->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    model->setContinuous(true);
    model->setStateModel([](
                            mpc::cvec<TVAR(Tnx)> &dx,
                            const mpc::cvec<TVAR(Tnx)> &x,
                            const mpc::cvec<TVAR(Tnu)> &u,
                            const unsigned int& p)
                        {
            dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
            dx[1] = x[0]; });

    conFunc->setModel(model, mapping);

    mpc::cvec<TVAR(Tnx)> x0;
    x0.resize(Tnx);
    x0 << 0, 0;
    conFunc->setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<TVAR(((Tph * Tnx) + (Tnu * Tch) + 1))> x;
    x.resize((Tph * Tnx) + (Tnu * Tch) + 1);
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    mpc::cvec<TVAR(Tph * Tnx)> costExpected;
    costExpected.resize(Tph * Tnx);
    costExpected << 0, -1, -2, -2, -2, -2, -2, -2, -2, -2;

    auto c = conFunc->evaluateStateModelEq(x, false);

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
    static constexpr auto sizer = mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(0), TVAR(Tny), TVAR(Tph), TVAR(Tch), TVAR(Tineq), TVAR(Teq));

    std::shared_ptr<mpc::Constraints<sizer>> conFunc;
    conFunc = std::make_shared<mpc::Constraints<sizer>>();
    conFunc->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    std::shared_ptr<mpc::Mapping<sizer>> mapping;
    mapping = std::make_shared<mpc::Mapping<sizer>>();
    mapping->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    std::shared_ptr<mpc::Model<sizer>> model;
    model = std::make_shared<mpc::Model<sizer>>();
    model->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    model->setContinuous(true);
    model->setStateModel([](
                            mpc::cvec<TVAR(Tnx)> &dx,
                            const mpc::cvec<TVAR(Tnx)>& x,
                            const mpc::cvec<TVAR(Tnu)>& u,
                            const unsigned int& p)
                        {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0]; });

    conFunc->setModel(model, mapping);

    conFunc->setIneqConstraints([](
                                   mpc::cvec<TVAR(Tineq)> &ieq_con,
                                   const mpc::mat<TVAR(Tph + 1), TVAR(Tnx)>& x,
                                   const mpc::mat<TVAR(Tph + 1), TVAR(Tny)>&,
                                   const mpc::mat<TVAR(Tph + 1), TVAR(Tnu)>&,
                                   const double&)
                               {
        for (int i = 0; i < Tineq; i++)
        {
            ieq_con[i] = x(0,0);
        } },1e-10);

    mpc::cvec<TVAR(Tnx)> x0;
    x0.resize(Tnx);
    x0 << 10, 0;
    conFunc->setCurrentState(x0);

    mpc::cvec<TVAR(((Tph * Tnx) + (Tnu * Tch) + 1))> x;
    x.resize((Tph * Tnx) + (Tnu * Tch) + 1);
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    mpc::cvec<TVAR(Tineq)> costExpected;
    costExpected.resize(Tineq);
    costExpected << x0[0];

    auto c = conFunc->evaluateIneq(x, false);

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
    static constexpr auto sizer = mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(0), TVAR(Tny), TVAR(Tph), TVAR(Tch), TVAR(Tineq), TVAR(Teq));

    std::shared_ptr<mpc::Constraints<sizer>> conFunc;
    conFunc = std::make_shared<mpc::Constraints<sizer>>();
    conFunc->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    std::shared_ptr<mpc::Mapping<sizer>> mapping;
    mapping = std::make_shared<mpc::Mapping<sizer>>();    
    mapping->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    std::shared_ptr<mpc::Model<sizer>> model;
    model = std::make_shared<mpc::Model<sizer>>();    
    model->initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    model->setContinuous(true);
    model->setStateModel([](
                            mpc::cvec<TVAR(Tnx)> &dx,
                            const mpc::cvec<TVAR(Tnx)>& x,
                            const mpc::cvec<TVAR(Tnu)>& u,
                            const unsigned int& p)
                        {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0]; });

    conFunc->setModel(model, mapping);

    conFunc->setEqConstraints([](
                                 mpc::cvec<TVAR(Teq)> &eq_con,
                                 const mpc::mat<TVAR(Tph + 1), TVAR(Tnx)>& x,
                                 const mpc::mat<TVAR(Tph + 1), TVAR(Tnu)>&)
                             {
        for (int i = 0; i < Teq; i++)
        {
            eq_con[i] = x(0, 0);
        } },1e-10);

    mpc::cvec<TVAR(Tnx)> x0;
    x0.resize(Tnx);
    x0 << 10, 0;
    conFunc->setCurrentState(x0);

    mpc::cvec<TVAR(((Tph * Tnx) + (Tnu * Tch) + 1))> x;
    x.resize((Tph * Tnx) + (Tnu * Tch) + 1);
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    mpc::cvec<TVAR(Teq)> costExpected;
    costExpected.resize(Teq);
    costExpected << x0[0];

    auto c = conFunc->evaluateEq(x, false);

    REQUIRE(c.value == costExpected);
    REQUIRE(c.grad.isZero());
}