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

    bool saveData = false;
    int maxIterations = 10000;
    double ts = 0.1;

    mpc::NLMPC<MPC_DYNAMIC_TEST_VARS(
        Tnx, Tnu, Tny,
        Tph, Tch,
        Tineq, Teq)>
        optsolver;

    optsolver.initialize(
        Tnx, Tnu, 0, Tny,
        Tph, Tch,
        Tineq, Teq);
        
    optsolver.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    optsolver.setContinuosTimeModel(ts);
    
    mpc::mat<MPC_DYNAMIC_TEST_VAR(Tnx), MPC_DYNAMIC_TEST_VAR(Tnx)> A(Tnx, Tnx);
    mpc::mat<MPC_DYNAMIC_TEST_VAR(Tnx), MPC_DYNAMIC_TEST_VAR(Tnu)> B(Tnx, Tnu);
    mpc::mat<MPC_DYNAMIC_TEST_VAR(Tny), MPC_DYNAMIC_TEST_VAR(Tnx)> C(Tny, Tnx);
    mpc::mat<MPC_DYNAMIC_TEST_VAR(Tny), MPC_DYNAMIC_TEST_VAR(Tnu)> D(Tny, Tnu);
    
    A << 1, 0,
         1, 1;
    B << 1,
         0;
    C << 0, 1;
    D << 0;

    auto stateEq = [=](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)>& dx,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> u)
    {
        dx = A*x + B*u; 
    };
    optsolver.setStateSpaceFunction(stateEq);

    auto outEq = [=](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tny)>& y,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> u)
    {
        y = C*x + D*u; 
    };

    auto objEq = [](
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnu)> u,
        double)
    {
        return x.array().square().sum() + u.array().square().sum();
    };
    optsolver.setObjectiveFunction(objEq);

    auto conIneq = [=](
        mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tineq)>& ineq,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnx)>,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tny)>,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnu)> u,
        double)
    {
        for (int i = 0; i <= Tph; i++)
        {
            ineq(i) = u(i) - 0.5;
            ineq(i + (Tph + 1)) = -u(i) - 7;
        }
    };
    optsolver.setIneqConFunction(conIneq);

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> modelX(Tnx);
    modelX << 10,
              0;

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnu)> modelU(Tnu);
    modelU << 0;

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tny)> modelY(Tny);
    modelY << 0;

    auto r = optsolver.getLastResult();

    std::ofstream yFile;
    std::ofstream xFile;
    std::ofstream uFile;
    std::ofstream tFile;

    if(saveData)
    {
        yFile.open("y.txt");
        xFile.open("x.txt");
        uFile.open("u.txt");
        tFile.open("t.txt");
    }

    for (int i=0; i<maxIterations; i++) 
    {
        if(saveData)
        {
            yFile << modelY.transpose() << std::endl;
            xFile << modelX.transpose() << std::endl;
            uFile << modelU.transpose() << std::endl;
            tFile << (double)i * ts << std::endl;
        }

        r = optsolver.step(modelX, modelU);

        modelU = r.cmd;
        outEq(modelY, modelX, modelU);
        stateEq(modelX, modelX, modelU);

        if (std::fabs(modelX(0)) <= 1e-4) 
        {
            break;
        }
    }

    if(saveData)
    {
        yFile.close();
        xFile.close();
        uFile.close();
        tFile.close();
    }
    return 0;
}

TEST_CASE(
    MPC_TEST_NAME("Discrete LTI SISO example"), 
    MPC_TEST_TAGS("[discrete][lti]"))
{
    REQUIRE(DiscreteLtiSiso() == 0);
}