#include "basic.hpp"
#include <catch2/catch.hpp>

#include <iostream>
#include <fstream>

int DiscreteLtiSiso()
{
    constexpr int Tnx = 2;
    constexpr int Tnu = 1;
    constexpr int Tny = 1;
    constexpr int Tph = 10;
    constexpr int Tch = 5;
    constexpr int Tineq = (Tph + 1) * 2;
    constexpr int Teq = 0;

    double ts = 0.1;

#ifdef MPC_DYNAMIC
    mpc::NLMPC<> optsolver(
        Tnx, Tnu, Tny,
        Tph, Tch,
        Tineq, Teq);
#else
    mpc::NLMPC<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> optsolver;
#endif
        
    optsolver.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    optsolver.setContinuosTimeModel(ts);
    
    mpc::mat<TVAR(Tnx), TVAR(Tnx)> A(Tnx, Tnx);
    mpc::mat<TVAR(Tnx), TVAR(Tnu)> B(Tnx, Tnu);
    mpc::mat<TVAR(Tny), TVAR(Tnx)> C(Tny, Tnx);
    mpc::mat<TVAR(Tny), TVAR(Tnu)> D(Tny, Tnu);
    
    A << 1, 0,
         1, 1;
    B << 1,
         0;
    C << 0, 1;
    D << 0;

    auto stateEq = [=](
        mpc::cvec<TVAR(Tnx)>& dx,
        mpc::cvec<TVAR(Tnx)> x,
        mpc::cvec<TVAR(Tnu)> u)
    {
        dx = A*x + B*u; 
    };
    optsolver.setStateSpaceFunction(stateEq);

    auto outEq = [=](
        mpc::cvec<TVAR(Tny)>& y,
        mpc::cvec<TVAR(Tnx)> x,
        mpc::cvec<TVAR(Tnu)> u)
    {
        y = C*x + D*u; 
    };

    optsolver.setOutputFunction(outEq);

    auto objEq = [](
        mpc::mat<TVAR(Tph + 1), TVAR(Tnx)> x,
        mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> u,
        double)
    {
        return x.array().square().sum() + u.array().square().sum();
    };
    optsolver.setObjectiveFunction(objEq);

    auto conIneq = [=](
        mpc::cvec<TVAR(Tineq)>& ineq,
        mpc::mat<TVAR(Tph + 1), TVAR(Tnx)>,
        mpc::mat<TVAR(Tph + 1), TVAR(Tny)>,
        mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> u,
        double)
    {
        for (int i = 0; i <= Tph; i++)
        {
            ineq(i) = u(i) - 0.5;
            ineq(i + (Tph + 1)) = -u(i) - 7;
        }
    };
    optsolver.setIneqConFunction(conIneq);

    mpc::cvec<TVAR(Tnx)> modelX(Tnx);
    modelX << 10,
              0;

    mpc::cvec<TVAR(Tnu)> modelU(Tnu);
    modelU << 0;

    mpc::cvec<TVAR(Tny)> modelY(Tny);
    modelY << 0;

    optsolver.getLastResult();
    return 0;
}

TEST_CASE(
    MPC_TEST_NAME("Discrete LTI SISO example"), 
    MPC_TEST_TAGS("[discrete][lti]"))
{
    REQUIRE(DiscreteLtiSiso() == 0);
}