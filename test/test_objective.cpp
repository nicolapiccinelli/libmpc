#include "basic.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking objective function"), 
    MPC_TEST_TAGS("[objective][template]"),
    ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
    (5, 3, 7, 7))
{
    static constexpr int Tny = 1;
    static constexpr int Tineq = 0;
    static constexpr int Teq = 0;

    mpc::ObjFunction<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> objFunc;
    objFunc.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    mpc::Mapping<MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq)> mapping;
    mapping.initialize(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq);

    objFunc.setMapping(mapping);
    objFunc.setContinuos(true);
    objFunc.setUserFunction([](
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnx)> x,
        mpc::mat<MPC_DYNAMIC_TEST_VAR(Tph + 1), MPC_DYNAMIC_TEST_VAR(Tnu)> u,
        double e) 
    {
        return x.array().square().sum() + u.array().square().sum();
    });

    mpc::cvec<MPC_DYNAMIC_TEST_VAR(Tnx)> x0;
    x0.resize(Tnx);
    x0 << 0, 0, 0, 0, 0;
    objFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<MPC_DYNAMIC_TEST_VAR((mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize)))> x, expectedGrad;
    x.resize(mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize));
    expectedGrad.resize(mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize));
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    double expectedValue = 65730.0;
    expectedGrad << 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 
    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 
    0, 1, 0, 0, 1, 0;

    auto c = objFunc.evaluate(x, false);
    REQUIRE(c.value == 65730.0);

    // mpc::cvec<MPC_DYNAMIC_TEST_VAR((mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize)))> abs_diff;
    // abs_diff.resize(mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(mpc::sizeEnum::DecVarsSize));
    // abs_diff = (c.grad - expectedGrad);
    // abs_diff = abs_diff.array().abs();
    // REQUIRE(abs_diff.maxCoeff() <= 1e-10);
}
