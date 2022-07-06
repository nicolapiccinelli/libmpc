#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking mapping dim"),
    MPC_TEST_TAGS("[mapping][template]"),
    ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch), 
    (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1), 
    (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    static constexpr int Tny = 1;
    static constexpr int Tineq = 1;
    static constexpr int Teq = 1;

    mpc::Mapping<mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), 0, TVAR(Tny), TVAR(Tph), TVAR(Tch), TVAR(Tineq), TVAR(Teq))> mapping;

    mapping.initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    REQUIRE(mapping.Iz2u().rows() == Tph * Tnu);
    REQUIRE(mapping.Iz2u().cols() == Tch * Tnu);

    REQUIRE(mapping.Iu2z().rows() == Tch * Tnu);
    REQUIRE(mapping.Iu2z().cols() == Tph * Tnu);
    
    REQUIRE(mapping.Sz2u().rows() == Tnu);
    REQUIRE(mapping.Sz2u().cols() == Tnu);

    REQUIRE(mapping.Su2z().rows() == Tnu);
    REQUIRE(mapping.Su2z().cols() == Tnu);

    REQUIRE(mapping.InputScaling().rows() == Tnu);
    REQUIRE(mapping.InputScaling().cols() == 1);

    REQUIRE(mapping.StateScaling().rows() == Tnx);
    REQUIRE(mapping.StateScaling().cols() == 1);

    REQUIRE(mapping.StateInverseScaling().rows() == Tnx);
    REQUIRE(mapping.StateInverseScaling().cols() == 1);
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking vector unwrapping"), 
    MPC_TEST_TAGS("[mapping][template]"),
    ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
    (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1),
    (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    static constexpr int Tny = 1;
    static constexpr int Tineq = 1;
    static constexpr int Teq = 1;

    mpc::Mapping<mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), 0, TVAR(Tny), TVAR(Tph), TVAR(Tch), TVAR(Tineq), TVAR(Teq))> mapping;
    mapping.initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    // input decision variables vector
    mpc::cvec<TVAR(((Tph * Tnx) + (Tnu * Tch) + 1))> x;
    x.resize((Tph * Tnx) + (Tnu * Tch) + 1);
    for (int i = 0; i < x.rows(); i++)
    {
        x[i] = i;
    }

    mpc::cvec<TVAR(Tnx)> x0;
    x0.resize(Tnx);
    for (int i = 0; i < x0.rows(); i++)
    {
        x0[i] = -i - 1;
    }

    // unwrapped components
    mpc::mat<TVAR(Tph + 1), TVAR(Tnx)> Xmat;
    mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> Umat;
    Xmat.resize(Tph + 1, Tnx);
    Umat.resize(Tph + 1, Tnu);
    double e;
    mapping.unwrapVector(x, x0, Xmat, Umat, e);

    // check row by row Xmat with the respected mock values
    REQUIRE(x0.transpose() == Xmat.row(0));
    for (int i = 1; i < Tph + 1; i++)
    {
        REQUIRE(x.middleRows((i - 1) * Tnx, Tnx).transpose() == Xmat.row(i));
    }

    // check row by row Umat with the respected mock values
    // once the control horizon ended the last optimal command
    // is replicated to fill the prediction horizon
    int u_index = 0;
    for (int i = 0; i < Tph + 1; i++)
    {
        if (i < Tch)
        {
            u_index = (Tph * Tnx) + (i * Tnu);
        }

        REQUIRE(x.middleRows(u_index, Tnu).transpose() == Umat.row(i));
    }

    // check slack value
    REQUIRE(x(x.size() - 1) == e);
}