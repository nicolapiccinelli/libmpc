#pragma once

#include <mpc/NLMPC.hpp>
#include <mpc/LMPC.hpp>

#ifdef MPC_DYNAMIC
#define MPC_DYNAMIC_TEST_NAME "Dynamic - "
#define MPC_DYNAMIC_TEST_TAGS "[dynamic]"
#define MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq) -1, -1, -1, -1, -1, -1, -1
#define MPC_LIN_DYNAMIC_TEST_VARS(Tnx, Tnu, Tndu, Tny, Tph, Tch) -1, -1, -1, -1, -1, -1
#define MPC_DYNAMIC_TEST_VAR(var) -1
#else
#define MPC_DYNAMIC_TEST_NAME "Static - "
#define MPC_DYNAMIC_TEST_TAGS "[static]"
#define MPC_DYNAMIC_TEST_VARS(Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq) Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq
#define MPC_LIN_DYNAMIC_TEST_VARS(Tnx, Tnu, Tndu, Tny, Tph, Tch) Tnx, Tnu, Tndu, Tny, Tph, Tch
#define MPC_DYNAMIC_TEST_VAR(var) var
#endif

#define MPC_TEST_NAME(name) MPC_DYNAMIC_TEST_NAME name
#define MPC_TEST_TAGS(tags) MPC_DYNAMIC_TEST_TAGS tags