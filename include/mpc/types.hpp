#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpc/logger.hpp>
#include <vector>

#if defined(debug) && defined(__linux__)
#include <cstdlib>
#include <iostream>
#include <stacktrace.hpp>
#include <stdexcept>

#undef eigen_assert
#define eigen_assert(x)                                      \
    if (!(x))                                                \
    {                                                        \
        Teuchos::show_stacktrace();                          \
        throw(std::runtime_error("Eigen assertion failed")); \
    }
#endif

#include <Eigen/Core>

namespace mpc
{
template <
    int M = Eigen::Dynamic,
    int N = Eigen::Dynamic>
    using mat = Eigen::Matrix<double, M, N>;

//template <
//    int M = Eigen::Dynamic,
//    int N = Eigen::Dynamic,
//    int K = Eigen::Dynamic>
//    using mat3 = std::array<Eigen::Matrix<double, M, N>, K>;

template <
    int N = Eigen::Dynamic>
    using cvec = Eigen::Matrix<double, N, 1>;

template <
    int N = Eigen::Dynamic>
    using rvec = Eigen::Matrix<double, 1, N>;

struct Parameters
{
    double relative_ftol;
    double relative_xtol;
    int maximum_iteration;
};

enum constraints_type
{
    INEQ,
    EQ,
    UINEQ,
    UEQ
};

} // namespace mpc
