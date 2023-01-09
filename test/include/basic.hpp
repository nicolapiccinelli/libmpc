/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/NLMPC.hpp>
#include <mpc/LMPC.hpp>

#ifdef MPC_DYNAMIC
#define MPC_DYNAMIC_TEST_NAME "Dynamic - "
#define MPC_DYNAMIC_TEST_TAGS "[dynamic]"
constexpr bool DYNAMIC_ALLOC = true;
#else
#define MPC_DYNAMIC_TEST_NAME "Static - "
#define MPC_DYNAMIC_TEST_TAGS "[static]"
constexpr bool DYNAMIC_ALLOC = false;
#endif

constexpr int TVAR(int v)
{
    int ret = -1;
    if (!DYNAMIC_ALLOC) {
        ret = v;
    }
    
    return ret;
}

#define MPC_TEST_NAME(name) MPC_DYNAMIC_TEST_NAME name
#define MPC_TEST_TAGS(tags) MPC_DYNAMIC_TEST_TAGS tags