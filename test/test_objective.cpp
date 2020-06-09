#include "basic.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG("Checking objective function", "[objective][template]",
                       ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
                       (5, 3, 7, 7))
{
    mpc::ObjFunction<Tnx, Tnu, Tph, Tch> objFunc;
    mpc::Common<Tnx, Tnu, Tph, Tch> mapping;

    objFunc.setMapping(mapping);
    objFunc.setContinuos(true);
    objFunc.setUserFunction([](mpc::mat<Tph + 1, Tnx> x,
                               mpc::mat<Tph + 1, Tnu> u,
                               double e) {
        return x.array().square().sum() + u.array().square().sum();
    });

    mpc::cvec<Tnx> x0;
    x0 << 0, 0, 0, 0, 0;
    objFunc.setCurrentState(x0);

    // input decision variables vector
    mpc::cvec<DecVarsSize> x, expectedGrad;
    for (size_t i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    double expectedValue = 65730.0;
    expectedGrad << 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 
    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 
    0, 1, 0, 0, 1, 0;

    typename mpc::ObjFunction<Tnx, Tnu, Tph, Tch>::Cost c = objFunc.evaluate(x, false);

    mpc::cvec<DecVarsSize> abs_diff = (c.grad - expectedGrad);
    abs_diff = abs_diff.array().abs();

    REQUIRE(c.value == 65730.0);
    REQUIRE(abs_diff.maxCoeff() <= 1e-10);
}
