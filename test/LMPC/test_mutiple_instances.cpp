/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

TEST_CASE(
    MPC_TEST_NAME("Linear multiple instances"),
    MPC_TEST_TAGS("[linear]"))
{
    mpc::Logger::instance().setLevel(mpc::Logger::log_level::DEEP);

    // MPC LATERAL CONTROLLER
    constexpr int num_states_1 = 8;
    constexpr int num_output_1 = 2;
    constexpr int num_inputs_1 = 2;
    constexpr int num_dinputs_1 = 5;

    constexpr int num_states_2 = 4;
    constexpr int num_output_2 = 1;
    constexpr int num_inputs_2 = 1;
    constexpr int num_dinputs_2 = 3;

    constexpr int pred_hor = 10;
    constexpr int ctrl_hor = 3;

    // MPC LATERAL CONTROLLER
#ifdef MPC_DYNAMIC
    mpc::LMPC<> latController(
        num_states_1, num_inputs_1, num_dinputs_1, num_output_1,
        pred_hor, ctrl_hor);
#else
    mpc::LMPC<
        TVAR(num_states_1), TVAR(num_inputs_1), TVAR(num_dinputs_1), TVAR(num_output_1),
        TVAR(pred_hor), TVAR(ctrl_hor)>
        latController;
#endif

    mpc::mat<num_states_1, num_states_1> Ad;
    Ad.setIdentity();
    mpc::mat<num_states_1, num_inputs_1> Bd;
    Bd.setZero();
    mpc::mat<num_output_1, num_states_1> Cd;
    Cd.setZero();

    latController.setStateSpaceModel(Ad, Bd, Cd);
    latController.getLastResult();

    // MPC LONGITUDINAL CONTROLLER
#ifdef MPC_DYNAMIC
    mpc::LMPC<> longController(
        num_states_1, num_inputs_1, num_dinputs_1, num_output_1,
        pred_hor, ctrl_hor);
#else
    mpc::LMPC<
        TVAR(num_states_1), TVAR(num_inputs_1), TVAR(num_dinputs_1), TVAR(num_output_1),
        TVAR(pred_hor), TVAR(ctrl_hor)>
        longController;
#endif

    longController.setStateSpaceModel(Ad, Bd, Cd);
    longController.getLastResult();

    // MPC VERTICAL CONTROLLER
#ifdef MPC_DYNAMIC
    mpc::LMPC<> vertController(
        num_states_2, num_inputs_2, num_dinputs_2, num_output_2,
        pred_hor, ctrl_hor);
#else
    mpc::LMPC<
        TVAR(num_states_2), TVAR(num_inputs_2), TVAR(num_dinputs_2), TVAR(num_output_2),
        TVAR(pred_hor), TVAR(ctrl_hor)>
        vertController;
#endif

    mpc::mat<num_states_2, num_states_2> Ad_2;
    Ad_2.setIdentity();
    mpc::mat<num_states_2, num_inputs_2> Bd_2;
    Bd_2.setZero();
    mpc::mat<num_output_2, num_states_2> Cd_2;
    Cd_2.setIdentity();

    vertController.setStateSpaceModel(Ad_2, Bd_2, Cd_2);
    vertController.getLastResult();
}