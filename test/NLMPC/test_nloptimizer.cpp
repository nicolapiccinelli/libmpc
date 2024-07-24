/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking state and input bounds"),
    MPC_TEST_TAGS("[nloptimizer][template]"),
    ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
    (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1),
    (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    // create the sizer
    static constexpr auto sizer = mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(0), TVAR(0), TVAR(Tph), TVAR(Tch), TVAR(0), TVAR(0));

    // create the nloptimizer
    std::shared_ptr<mpc::NLOptimizer<sizer>> nlopt;
    nlopt = std::make_shared<mpc::NLOptimizer<sizer>>();
    nlopt->initialize(Tnx, Tnu, 0, 0, Tph, Tch, 0, 0);
    nlopt->onInit();

    std::cout << "Initialized the nloptimizer" << std::endl;

    // set the state bounds
    mpc::cvec<TVAR(Tnx)> xlb, xub;
    xlb.resize(Tnx);
    xlb.setConstant(-1.0);
    xub.resize(Tnx);
    xub.setConstant(1.0);

    nlopt->setStateBounds(xlb, xub, {-1, -1});

    std::cout << "Set the state bounds" << std::endl;

    // set the input bounds
    mpc::cvec<TVAR(Tnu)> ulb, uub;
    ulb.resize(Tnu);
    ulb.setConstant(-1.0);
    uub.resize(Tnu);
    uub.setConstant(1.0);

    nlopt->setInputBounds(ulb, uub, {-1, -1});

    std::cout << "Set the input bounds" << std::endl;

    // check if the bounds are set correctly
    auto lb_check = nlopt->getLowerBound();
    auto ub_check = nlopt->getUpperBound();

    std::cout << "Got the lower bound" << std::endl;
    std::cout << lb_check << std::endl;
    std::cout << "Got the upper bound" << std::endl;
    std::cout << ub_check << std::endl;

    for (int i = 0; i < Tph; i++)
    {
        for (int j = 0; j < Tnx; j++)
        {
            REQUIRE(lb_check[(i * Tnx) + j] == -1.0);
            REQUIRE(ub_check[(i * Tnx) + j] == 1.0);
        }
    }

    for (int i = 0; i < Tch; i++)
    {
        for (int j = 0; j < Tnu; j++)
        {
            REQUIRE(lb_check[(Tph * Tnx) + ((i * Tnu) + j)] == -1.0);
            REQUIRE(ub_check[(Tph * Tnx) + ((i * Tnu) + j)] == 1.0);
        }
    }
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking default state and input bounds"),
    MPC_TEST_TAGS("[nloptimizer][template]"),
    ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
    (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1),
    (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    // create the sizer
    static constexpr auto sizer = mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), TVAR(0), TVAR(0), TVAR(Tph), TVAR(Tch), TVAR(0), TVAR(0));

    // create the nloptimizer
    std::shared_ptr<mpc::NLOptimizer<sizer>> nlopt;
    nlopt = std::make_shared<mpc::NLOptimizer<sizer>>();
    nlopt->initialize(Tnx, Tnu, 0, 0, Tph, Tch, 0, 0);
    nlopt->onInit();

    std::cout << "Initialized the nloptimizer" << std::endl;

    // check if the bounds are set correctly
    auto lb_check = nlopt->getLowerBound();
    auto ub_check = nlopt->getUpperBound();

    std::cout << "Got the lower bound" << std::endl;
    std::cout << lb_check << std::endl;
    std::cout << "Got the upper bound" << std::endl;
    std::cout << ub_check << std::endl;

    for (int i = 0; i < Tph; i++)
    {
        for (int j = 0; j < Tnx; j++)
        {
            REQUIRE(lb_check[(i * Tnx) + j] == -std::numeric_limits<double>::infinity());
            REQUIRE(ub_check[(i * Tnx) + j] == std::numeric_limits<double>::infinity());
        }
    }

    for (int i = 0; i < Tch; i++)
    {
        for (int j = 0; j < Tnu; j++)
        {
            REQUIRE(lb_check[(Tph * Tnx) + ((i * Tnu) + j)] == -std::numeric_limits<double>::infinity());
            REQUIRE(ub_check[(Tph * Tnx) + ((i * Tnu) + j)] == std::numeric_limits<double>::infinity());
        }
    }
}