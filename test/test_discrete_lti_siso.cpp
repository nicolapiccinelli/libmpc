#include "basic.hpp"
#include <catch2/catch.hpp>

int DiscreteLtiSiso()
{
    constexpr int Tnx = 1;
    constexpr int Tnu = 1;
    constexpr int Tny = 1;
    constexpr int Tph = 10;
    constexpr int Tch = 5;
    constexpr int Tineq = (Tph + 1)*2;
    constexpr int Teq = 0;

    int maxIterations = 10000;

    double ts = 0.1;

    bool useHardConst = false;

    mpc::NLMPC<MPC_DYNAMIC_TEST_VARS(
        Tnx, Tnu, Tny,
        Tph, Tch,
        Tineq, Teq)>
        optsolver;

    optsolver.initialize(
        useHardConst,
        Tnx, Tnu, Tny,
        Tph, Tch,
        Tineq, Teq);
    optsolver.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    optsolver.setSampleTime(ts);

    double A = 1.5;
    double B = 1;
    double C = 1;
    double D = 0;

    auto stateEq = [&](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)>& dx,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> u)
    {
        dx = A*x + B*u; 
    };
    optsolver.setStateSpaceFunction(stateEq);

    auto outEq = [&](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tny)>& y,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> u)
    {
        y = C*x + D*u; 
    };

    auto objEq = [&](
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnu)> u,
        double e)
    {
        return x.array().square().sum() + u.array().square().sum();
    };
    optsolver.setObjectiveFunction(objEq);

    auto conIneq = [&](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tineq)>& ineq,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnu)> u,
        double e)
    {
        for (int i = 0; i <= Tph; i++)
        {
            ineq(i) = u(i) - 0.5;
            ineq(i+Tph+1) = -u(i) - 7;
        }
    };
    optsolver.setIneqConFunction(conIneq);

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> modelX;
    modelX.resize(Tnx);
    modelX(0) = 10;

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> modelU;
    modelU.resize(Tnu);
    modelU(0) = 0;

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> modelY;
    modelY.resize(Tny);
    modelY(0) = 0;

    auto r = optsolver.getLastResult();

    for (int i=1; i<maxIterations; i++) 
    {
        r = optsolver.step(modelX, modelU);

        modelU = r.cmd;
        outEq(modelY, modelX, modelU);
        stateEq(modelX, modelX, modelU);

        if (std::fabs(modelX(0)) <= 1e-4) 
        {
            break;
        }
    }

    return 0;
}

TEST_CASE(
    MPC_TEST_NAME("Discrete LTI SISO example"), 
    MPC_TEST_TAGS("[discrete][lti]"))
{
    REQUIRE(DiscreteLtiSiso() == 0);
}