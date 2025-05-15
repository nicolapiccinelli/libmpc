/*
 *   Copyright (c) 2025 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <Eigen/Dense>

#define NLMPC_TEMPLATE_PARAMS ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq, int Teq), Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)
#define NLMPC_TEMPLATE_CASES \
    (1, 1, 1, 1, 1, 0, 0), \
    (5, 1, 1, 1, 1, 2, 0), \
    (5, 3, 1, 1, 1, 0, 2), \
    (5, 3, 1, 7, 1, 2, 2), \
    (5, 3, 1, 7, 4, 4, 8), \
    (5, 3, 1, 7, 7, 10, 5)

template <int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq, int Teq>
auto make_solver() {
    #ifdef MPC_DYNAMIC
        return mpc::NLMPC<>(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);
    #else
        return mpc::NLMPC<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    #endif
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setDiscretizationSamplingTime"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    double ts = 0.1;
    REQUIRE(optsolver.setDiscretizationSamplingTime(ts) == true);
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setInputScale"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tnu, 1> scaling = Eigen::Matrix<double, Tnu, 1>::Ones();
    REQUIRE_NOTHROW(optsolver.setInputScale(scaling));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setStateScale"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tnx, 1> scaling = Eigen::Matrix<double, Tnx, 1>::Ones();
    REQUIRE_NOTHROW(optsolver.setStateScale(scaling));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setObjectiveFunction"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    auto obj_fun = [](const mpc::mat<TVAR(Tph + 1), TVAR(Tnx)> &,
                      const mpc::mat<TVAR(Tph + 1), TVAR(Tny)> &,
                      const mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> &,
                      const double &) -> double
    { return 0.0; };
    REQUIRE(optsolver.setObjectiveFunction(obj_fun));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setStateSpaceFunction"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    auto state_fun = [](mpc::cvec<TVAR(Tnx)> &dx,
                        const mpc::cvec<TVAR(Tnx)> &,
                        const mpc::cvec<TVAR(Tnu)> &,
                        int)
    {
        dx.setZero();
    };
    REQUIRE(optsolver.setStateSpaceFunction(state_fun));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setOutputFunction"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    auto out_fun = [](mpc::cvec<TVAR(Tny)> &y,
                      const mpc::cvec<TVAR(Tnx)> &,
                      const mpc::cvec<TVAR(Tnu)> &,
                      int)
    {
        y.setZero();
    };
    REQUIRE(optsolver.setOutputFunction(out_fun));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setIneqConFunction"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    if constexpr (Tineq > 0)
    {
        auto ineq_fun = [](mpc::cvec<TVAR(Tineq)> &ineq,
                           const mpc::mat<TVAR(Tph + 1), TVAR(Tnx)> &,
                           const mpc::mat<TVAR(Tph + 1), TVAR(Tny)> &,
                           const mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> &,
                           const double &)
        {
            ineq.setZero();
        };
        REQUIRE(optsolver.setIneqConFunction(ineq_fun));
    }
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setEqConFunction"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    if constexpr (Teq > 0)
    {
        auto eq_fun = [](mpc::cvec<TVAR(Teq)> &eq,
                         const mpc::mat<TVAR(Tph + 1), TVAR(Tnx)> &,
                         const mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> &)
        {
            eq.setZero();
        };
        REQUIRE(optsolver.setEqConFunction(eq_fun));
    }
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setStateBounds_matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tnx, Tph> XMinMat = Eigen::Matrix<double, Tnx, Tph>::Constant(-1.0);
    Eigen::Matrix<double, Tnx, Tph> XMaxMat = Eigen::Matrix<double, Tnx, Tph>::Constant(1.0);
    REQUIRE(optsolver.setStateBounds(XMinMat, XMaxMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setInputBounds_matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tnu, Tch> UMinMat = Eigen::Matrix<double, Tnu, Tch>::Constant(-1.0);
    Eigen::Matrix<double, Tnu, Tch> UMaxMat = Eigen::Matrix<double, Tnu, Tch>::Constant(1.0);
    REQUIRE(optsolver.setInputBounds(UMinMat, UMaxMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setOutputBounds_matrix"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tny, Tph> YMinMat = Eigen::Matrix<double, Tny, Tph>::Constant(-1.0);
    Eigen::Matrix<double, Tny, Tph> YMaxMat = Eigen::Matrix<double, Tny, Tph>::Constant(1.0);

    REQUIRE_THROWS(optsolver.setOutputBounds(YMinMat, YMaxMat));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setStateBounds_slice"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tnx, 1> XMin = Eigen::Matrix<double, Tnx, 1>::Constant(-1.0);
    Eigen::Matrix<double, Tnx, 1> XMax = Eigen::Matrix<double, Tnx, 1>::Constant(1.0);

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setStateBounds(XMin, XMax, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setInputBounds_slice"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tnu, 1> UMin = Eigen::Matrix<double, Tnu, 1>::Constant(-1.0);
    Eigen::Matrix<double, Tnu, 1> UMax = Eigen::Matrix<double, Tnu, 1>::Constant(1.0);

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(optsolver.setInputBounds(UMin, UMax, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setOutputBounds_slice"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    Eigen::Matrix<double, Tny, 1> YMin = Eigen::Matrix<double, Tny, 1>::Constant(-1.0);
    Eigen::Matrix<double, Tny, 1> YMax = Eigen::Matrix<double, Tny, 1>::Constant(1.0);

    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE_THROWS(optsolver.setOutputBounds(YMin, YMax, slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC setOptimizerParameters"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    mpc::NLParameters params;
    REQUIRE_NOTHROW(optsolver.setOptimizerParameters(params));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("NLMPC logger_and_slice_methods"),
    MPC_TEST_TAGS("[interface][template]"),
    NLMPC_TEMPLATE_PARAMS,
    NLMPC_TEMPLATE_CASES)
{
    auto optsolver = make_solver<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
    REQUIRE_NOTHROW(optsolver.setLoggerLevel(mpc::Logger::LogLevel::NONE));
    REQUIRE_NOTHROW(optsolver.setLoggerPrefix("test"));
    REQUIRE_NOTHROW(optsolver.getLastResult());

    mpc::HorizonSlice slice{0, 0};
    REQUIRE_NOTHROW(optsolver.isSliceUnset(slice));
    REQUIRE_NOTHROW(optsolver.isPredictionHorizonSliceValid(slice));
    REQUIRE_NOTHROW(optsolver.isControlHorizonSliceValid(slice));
}