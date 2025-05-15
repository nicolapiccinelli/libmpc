/*
 *   Copyright (c) 2023-2025 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#define LMPC_TEMPLATE_PARAMS ((int Tnx, int Tnu, int Tndu, int Tny, int Tph, int Tch), Tnx, Tnu, Tndu, Tny, Tph, Tch)
#define LMPC_TEMPLATE_CASES \
        (1, 1, 1, 1, 1, 1), \
        (5, 1, 1, 1, 1, 1), \
        (5, 3, 3, 1, 1, 1), \
        (5, 3, 3, 7, 7, 7), \
        (5, 3, 3, 7, 4, 4), \
        (5, 3, 3, 7, 7, 5)

template <int Tnx, int Tnu, int Tndu, int Tny, int Tph, int Tch>
auto make_solver()
{
#ifdef MPC_DYNAMIC
    return mpc::LMPC<>(Tnx, Tnu, Tndu, Tny, Tph, Tch);
#else
    return mpc::LMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
#endif
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setStateBounds matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tnx, Tph> XMinMat = mpc::mat<Tnx, Tph>::Constant(-1.0);
    mpc::mat<Tnx, Tph> XMaxMat = mpc::mat<Tnx, Tph>::Constant(1.0);
    REQUIRE(optsolver.setStateBounds(XMinMat, XMaxMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setInputBounds matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tnu, Tch> UMinMat = mpc::mat<Tnu, Tch>::Constant(-1.0);
    mpc::mat<Tnu, Tch> UMaxMat = mpc::mat<Tnu, Tch>::Constant(1.0);
    REQUIRE(optsolver.setInputBounds(UMinMat, UMaxMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setOutputBounds matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tny, Tph> YMinMat = mpc::mat<Tny, Tph>::Constant(-1.0);
    mpc::mat<Tny, Tph> YMaxMat = mpc::mat<Tny, Tph>::Constant(1.0);
    REQUIRE(optsolver.setOutputBounds(YMinMat, YMaxMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setStateBounds slice"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::cvec<Tnx> XMin = mpc::cvec<Tnx>::Constant(-1.0);
    mpc::cvec<Tnx> XMax = mpc::cvec<Tnx>::Constant(1.0);
    
    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setStateBounds(XMin, XMax, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setInputBounds slice"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::cvec<Tnu> UMin = mpc::cvec<Tnu>::Constant(-1.0);
    mpc::cvec<Tnu> UMax = mpc::cvec<Tnu>::Constant(1.0);

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setInputBounds(UMin, UMax, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setOutputBounds slice"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::cvec<Tny> YMin = mpc::cvec<Tny>::Constant(-1.0);
    mpc::cvec<Tny> YMax = mpc::cvec<Tny>::Constant(1.0);

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setOutputBounds(YMin, YMax, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setObjectiveWeights matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tny, Tph> OWeightMat = mpc::mat<Tny, Tph>::Ones();
    mpc::mat<Tnu, Tph> UWeightMat = mpc::mat<Tnu, Tph>::Ones();
    mpc::mat<Tndu, Tph> DeltaUWeightMat = mpc::mat<Tndu, Tph>::Ones();
    REQUIRE(optsolver.setObjectiveWeights(OWeightMat, UWeightMat, DeltaUWeightMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setObjectiveWeights vector"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::cvec<Tny> OWeight = mpc::cvec<Tny>::Ones();
    mpc::cvec<Tnu> UWeight = mpc::cvec<Tnu>::Ones();
    mpc::cvec<Tndu> DeltaUWeight = mpc::cvec<Tndu>::Ones();

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setObjectiveWeights(OWeight, UWeight, DeltaUWeight, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setStateSpaceModel"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tnx, Tnx> A = mpc::mat<Tnx, Tnx>::Identity();
    mpc::mat<Tnx, Tnu> B = mpc::mat<Tnx, Tnu>::Zero();
    mpc::mat<Tny, Tnx> C = mpc::mat<Tny, Tnx>::Identity();
    REQUIRE(optsolver.setStateSpaceModel(A, B, C));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setDisturbances"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tnx, Tndu> Bd = mpc::mat<Tnx, Tndu>::Zero();
    mpc::mat<Tny, Tndu> Dd = mpc::mat<Tny, Tndu>::Zero();
    REQUIRE(optsolver.setDisturbances(Bd, Dd));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setExogenousInputs matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tndu, Tph> uMeasMat = mpc::mat<Tndu, Tph>::Zero();
    REQUIRE(optsolver.setExogenousInputs(uMeasMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setExogenousInputs vector"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::cvec<Tndu> uMeas = mpc::cvec<Tndu>::Zero();

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setExogenousInputs(uMeas, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setReferences matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::mat<Tny, Tph> outRefMat = mpc::mat<Tny, Tph>::Zero();
    mpc::mat<Tnu, Tph> cmdRefMat = mpc::mat<Tnu, Tph>::Zero();
    mpc::mat<Tndu, Tph> deltaCmdRefMat = mpc::mat<Tndu, Tph>::Zero();
    REQUIRE(optsolver.setReferences(outRefMat, cmdRefMat, deltaCmdRefMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setReferences vector"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    mpc::cvec<Tny> outRef = mpc::cvec<Tny>::Zero();
    mpc::cvec<Tnu> cmdRef = mpc::cvec<Tnu>::Zero();
    mpc::cvec<Tndu> deltaCmdRef = mpc::cvec<Tndu>::Zero();

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setReferences(outRef, cmdRef, deltaCmdRef, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC getSolverWarmStartPrimal"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    auto res = optsolver.getSolverWarmStartPrimal();
    REQUIRE(res.size() >= 0);
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC getSolverWarmStartDual"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    auto res = optsolver.getSolverWarmStartDual();
    REQUIRE(res.size() >= 0);
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("LMPC setSolverWarmStart"),
    MPC_TEST_TAGS("[interface][template]"),
    LMPC_TEMPLATE_PARAMS,
    LMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
    std::vector<double> warm_primal(10, 0.0);
    std::vector<double> warm_dual(10, 0.0);
    REQUIRE_NOTHROW(optsolver.setSolverWarmStart(warm_primal, warm_dual));
}
