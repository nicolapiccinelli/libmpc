/*
 *   Copyright (c) 2023-2026 Nicola Piccinelli
 *   All rights reserved.
 */
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

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking horizon slicing validation"),
    MPC_TEST_TAGS("[horizon slicing][template]"),
    ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
    (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1),
    (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    // instantiate the nlmpc class
    const size_t Tny = Tnx;
    mpc::NLMPC<> nlmpc(Tnx, Tnu, Tny, Tph, Tch, 0, 0);
    mpc::LMPC<> lmpc(Tnx, Tnu, 0, Tny, Tph, Tch);

    // test various horizon slice
    mpc::HorizonSlice slice = mpc::HorizonSlice::all();
    REQUIRE(nlmpc.isSliceUnset(slice));
    REQUIRE_FALSE(nlmpc.isPredictionHorizonSliceValid(slice));
    REQUIRE_FALSE(nlmpc.isControlHorizonSliceValid(slice));

    REQUIRE(lmpc.isSliceUnset(slice));
    REQUIRE_FALSE(lmpc.isPredictionHorizonSliceValid(slice));
    REQUIRE_FALSE(lmpc.isControlHorizonSliceValid(slice));

    slice.start = 0;
    slice.end = Tph;

    REQUIRE_FALSE(nlmpc.isSliceUnset(slice));
    REQUIRE(nlmpc.isPredictionHorizonSliceValid(slice));

    REQUIRE_FALSE(lmpc.isSliceUnset(slice));
    REQUIRE(lmpc.isPredictionHorizonSliceValid(slice));

    slice.start = 0;
    slice.end = Tch;

    REQUIRE_FALSE(nlmpc.isSliceUnset(slice));
    REQUIRE(nlmpc.isControlHorizonSliceValid(slice));

    REQUIRE_FALSE(lmpc.isSliceUnset(slice));
    REQUIRE(lmpc.isControlHorizonSliceValid(slice));

    slice.start = 0;
    slice.end = std::min(Tph, Tch);

    REQUIRE_FALSE(nlmpc.isSliceUnset(slice));
    REQUIRE(nlmpc.isPredictionHorizonSliceValid(slice));
    REQUIRE(nlmpc.isControlHorizonSliceValid(slice));

    REQUIRE_FALSE(lmpc.isSliceUnset(slice));
    REQUIRE(lmpc.isPredictionHorizonSliceValid(slice));
    REQUIRE(lmpc.isControlHorizonSliceValid(slice));

    slice.start = 0;
    slice.end = Tph + 1;

    REQUIRE_FALSE(nlmpc.isSliceUnset(slice));
    REQUIRE_FALSE(nlmpc.isPredictionHorizonSliceValid(slice));
    REQUIRE_FALSE(nlmpc.isControlHorizonSliceValid(slice));

    REQUIRE_FALSE(lmpc.isSliceUnset(slice));
    REQUIRE_FALSE(lmpc.isPredictionHorizonSliceValid(slice));
    REQUIRE_FALSE(lmpc.isControlHorizonSliceValid(slice));
}

TEMPLATE_TEST_CASE_SIG(
    MPC_TEST_NAME("Checking wrap/unwrap inverse consistency"),
    MPC_TEST_TAGS("[mapping][template][roundtrip]"),
    ((int Tnx, int Tnu, int Tph, int Tch), Tnx, Tnu, Tph, Tch),
    (1, 1, 1, 1), (5, 1, 1, 1), (5, 3, 1, 1),
    (5, 3, 7, 1), (5, 3, 7, 4), (5, 3, 7, 7))
{
    static constexpr int Tny = 1;
    static constexpr int Tineq = 1;
    static constexpr int Teq = 1;

    mpc::Mapping<mpc::MPCSize(TVAR(Tnx), TVAR(Tnu), 0, TVAR(Tny), TVAR(Tph), TVAR(Tch), TVAR(Tineq), TVAR(Teq))> mapping;
    mapping.initialize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq);

    // --- Create deterministic test data ---
    mpc::mat<TVAR(Tph + 1), TVAR(Tnx)> Xmat_ref{};
    mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> Umat_ref{};

    Xmat_ref.resize(Tph + 1, Tnx);
    Umat_ref.resize(Tph + 1, Tnu);

    // Fill Xmat
    for (int i = 0; i < Xmat_ref.rows(); ++i)
    {
        for (int j = 0; j < Xmat_ref.cols(); ++j)
        {
            Xmat_ref(i, j) = static_cast<double>(i) + static_cast<double>(j) * 0.1;
        }
    }

    // *** FIX: Create Umat_ref that *already* respects the "hold" condition ***
    for (int i = 0; i < Umat_ref.rows(); ++i)
    {
        for (int j = 0; j < Umat_ref.cols(); ++j)
        {
            // Determine the correct time index to use
            // If i < Tch, use 'i'. If i >= Tch, use 'Tch - 1'.
            int t_hold = std::min(i, Tch - 1);
            if (Tch == 0)
                t_hold = 0; // Edge case for Tch=0
            if (i == Tph)
                t_hold = Tph - 1; // Last row is a copy of Tph-1
            if (Tph == 0)
                t_hold = 0; // Edge case for Tph=0

            // Apply the hold logic to the reference data generation
            if (i < Tph)
            {
                int t = std::min(i, Tch - 1);
                if (Tch == 0)
                    t = 0; // Should not happen if Tch >= 1

                Umat_ref(i, j) = static_cast<double>(t) * 0.2 + static_cast<double>(j) * 0.3;
            }
            else // Last row i == Tph
            {
                if (Tph > 0)
                {
                    Umat_ref.row(Tph) = Umat_ref.row(Tph - 1);
                }
                else
                {
                    Umat_ref(i, j) = 0.0; // Tph=0, Umat is 1xNu, fill with 0
                }
            }
        }
    }
    // Now Umat_ref might look like [u0, u1, u2, u3, u3, u3, u3] (for Tch=4)

    mpc::cvec<TVAR(Tnx)> x0_ref = Xmat_ref.row(0).transpose();
    double slack_ref{0.42};

    // --- Wrap then unwrap ---
    auto z = mapping.wrapVector(Xmat_ref, Umat_ref, slack_ref);

    mpc::mat<TVAR(Tph + 1), TVAR(Tnx)> Xmat_unwrap{};
    mpc::mat<TVAR(Tph + 1), TVAR(Tnu)> Umat_unwrap{};
    double slack_unwrap{};

    // Resize to prevent crash
    Xmat_unwrap.resize(Tph + 1, Tnx);
    Umat_unwrap.resize(Tph + 1, Tnu);

    mapping.unwrapVector(z, x0_ref, Xmat_unwrap, Umat_unwrap, slack_unwrap);

    // --- Validate consistency ---

    // State consistency
    REQUIRE(Xmat_unwrap.row(0).isApprox(x0_ref.transpose(), 1e-10));
    for (int i = 1; i < Xmat_ref.rows(); ++i)
    {
        REQUIRE(Xmat_unwrap.row(i).isApprox(Xmat_ref.row(i), 1e-10));
    }

    // *** Control consistency (Your original checks will now pass) ***
    for (int i = 0; i < Tph; ++i)
    {
        REQUIRE(Umat_unwrap.row(i).isApprox(Umat_ref.row(i), 1e-10));
    }
    if (Tph > 0)
    {
        REQUIRE(Umat_unwrap.row(Tph).isApprox(Umat_ref.row(Tph - 1), 1e-10));
    }

    // Slack consistency
    REQUIRE(std::abs(slack_unwrap - slack_ref) < 1e-12);
}