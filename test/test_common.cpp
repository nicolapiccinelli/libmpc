#include "basic.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG("Checking mapping dimensions", "[mapping][template]",
                       ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch), 
                       (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1), 
                       (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    mpc::Common<Tnx, Tnu, Tph, Tch> mapping;

    REQUIRE(mapping.Iz2u.rows() == Tph * Tnu);
    REQUIRE(mapping.Iz2u.cols() == Tch * Tnu);

    REQUIRE(mapping.Iu2z.rows() == Tch * Tnu);
    REQUIRE(mapping.Iu2z.cols() == Tph * Tnu);

    REQUIRE(mapping.Sz2u.rows() == Tnu);
    REQUIRE(mapping.Sz2u.cols() == Tnu);

    REQUIRE(mapping.Su2z.rows() == Tnu);
    REQUIRE(mapping.Su2z.cols() == Tnu);
}

TEMPLATE_TEST_CASE_SIG("Checking vector unwrapping", "[mapping][template]",
                       ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
                       (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1),
                       (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    mpc::Common<Tnx, Tnu, Tph, Tch> mapping;

    // input decision variables vector
    mpc::cvec<DecVarsSize> x;
    mpc::cvec<Tnx> x0;
    for (size_t i = 0; i < x.rows() ; i++)
    {
        x[i] = i;
    }

    for (size_t i = 0; i < x0.rows(); i++)
    {
        x0[i] = -i - 1;
    }

    // unwrapped components
    mpc::mat<Tph + 1, Tnx> Xmat;
    mpc::mat<Tph + 1, Tnu> Umat;
    double e;
    mapping.unwrapVector(x, x0, Xmat, Umat, e);

    // check row by row Xmat with the respected mock values
    REQUIRE(x0.transpose() == Xmat.row(0));
    for (size_t i = 1; i < Tph + 1; i++)
    {
        REQUIRE(x.middleRows((i - 1) * Tnx, Tnx).transpose() == Xmat.row(i));
    }

    // check row by row Umat with the respected mock values
    // once the control horizon ended the last optimal command
    // is replicated to fill the prediction horizon
    int u_index = 0;
    for (size_t i = 0; i < Tph + 1; i++)
    {
        if (i < Tch)
        {
            u_index = (Tph * Tnx) + (i * Tnu);
        }

        REQUIRE(x.middleRows(u_index, Tnu).transpose() == Umat.row(i));
    }

    // check slack value
    REQUIRE(x[x.size() - 1] == e);
}