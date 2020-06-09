#include "basic.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG("Checking missing user constraints", "[constraints][template]",
                       ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
                       (1, 1, 1, 1, 1, 0), (5, 1, 1, 1, 1, 0), (5, 3, 1, 1, 1, 0),
                       (5, 3, 1, 7, 1, 0), (5, 3, 1, 7, 4, 0), (5, 3, 1, 7, 7, 0))
{
    constexpr int Teq = Tineq;

    mpc::ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conFunc;

    REQUIRE_FALSE(conFunc.hasEqConstraintFunction());
    REQUIRE_FALSE(conFunc.hasIneqConstraintFunction());
}

TEMPLATE_TEST_CASE_SIG("Checking model equality constraints", "[constraints][template]",
                       ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
                       (2, 1, 1, 5, 5, 0))
{
    constexpr int Teq = 0;
    mpc::ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conFunc;
    mpc::Common<Tnx, Tnu, Tph, Tch> mapping;

    conFunc.setMapping(mapping);
    conFunc.setContinuos(true);
    conFunc.setStateSpaceFunction([](mpc::cvec<Tnx> &dx,
                                     mpc::cvec<Tnx> x,
                                     mpc::cvec<Tnu> u) {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
    });

    mpc::cvec<Tnx> x0;
    x0 << 0, 0;
    conFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<DecVarsSize> x;
    for (size_t i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    mpc::cvec<StateEqSize> costExpected;
    costExpected << 0, -1, -2, -2, -2, -2, -2, -2, -2, -2;

    typename mpc::ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::template Cost<StateEqSize>
        c = conFunc.evaluateEq(x, false);

    REQUIRE(c.value == costExpected);
    REQUIRE(c.grad.isZero());
}

TEMPLATE_TEST_CASE_SIG("Checking user inequality constraints", "[constraints][template]",
                       ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
                       (2, 1, 1, 5, 5, 1))
{
    constexpr int Teq = 0;
    mpc::ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conFunc;
    mpc::Common<Tnx, Tnu, Tph, Tch> mapping;

    conFunc.setMapping(mapping);
    conFunc.setContinuos(true);
    conFunc.setStateSpaceFunction([](mpc::cvec<Tnx> &dx,
                                     mpc::cvec<Tnx> x,
                                     mpc::cvec<Tnu> u) {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
    });

    conFunc.setIneqConstraintFunction([](mpc::cvec<Tineq> &eq_con,
                                              mpc::mat<Tph + 1, Tnx> x,
                                              mpc::mat<Tph + 1, Tnu> u,
                                              double e) {
        for (size_t i = 0; i < Tineq; i++)
        {
            eq_con[i] = x(0,0);
        }
    });

    mpc::cvec<Tnx> x0;
    x0 << 10, 0;
    conFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<DecVarsSize> x;
    for (size_t i = 0; i < x.rows(); i++)
    {
        x[i] = i + 1;
    }

    mpc::cvec<Tineq> costExpected;
    costExpected << x0[0];

    typename mpc::ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::template Cost<Tineq>
        c = conFunc.evaluateUserIneq(x, false);

    REQUIRE(c.value == costExpected);
    REQUIRE(c.grad.isZero());
}

TEMPLATE_TEST_CASE_SIG("Checking user equality constraints", "[constraints][template]",
                       ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq), Tnx, Tnu, Tny, Tph, Tch, Tineq),
                       (2, 1, 1, 5, 5, 0))
{
    constexpr int Teq = 1;
    mpc::ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conFunc;
    mpc::Common<Tnx, Tnu, Tph, Tch> mapping;

    conFunc.setMapping(mapping);
    conFunc.setContinuos(true);
    conFunc.setStateSpaceFunction([](mpc::cvec<Tnx> &dx,
                                     mpc::cvec<Tnx> x,
                                     mpc::cvec<Tnu> u) {
        dx[0] = ((1.0 - (x[1] * x[1])) * x[0]) - x[1] + u[0];
        dx[1] = x[0];
    });

    conFunc.setEqConstraintFunction([Teq](mpc::cvec<Teq> &eq_con,
                                          mpc::mat<Tph + 1, Tnx> x,
                                          mpc::mat<Tph + 1, Tnu> u) {
        for (size_t i = 0; i < Teq; i++)
        {
            eq_con[i] = x(0, 0);
        }
    });

    mpc::cvec<Tnx> x0;
    x0 << 10, 0;
    conFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<DecVarsSize> x;
    for (size_t i = 0; i < x.rows(); i++)
    {
        x[i] = i + 1;
    }

    mpc::cvec<Teq> costExpected;
    costExpected << x0[0];

    typename mpc::ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::template Cost<Teq>
        c = conFunc.evaluateUserEq(x, false);

    REQUIRE(c.value == costExpected);
    REQUIRE(c.grad.isZero());
}