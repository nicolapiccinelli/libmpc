#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpc/logger.hpp>
#include <vector>

#if debug && defined(__linux__)
#include <cstdlib>
#include <iostream>
#include <stacktrace.hpp>
#include <stdexcept>

#undef eigen_assert
#define eigen_assert(x)                                      \
    if (!(x)) {                                              \
        Teuchos::show_stacktrace();                          \
        throw(std::runtime_error("Eigen assertion failed")); \
    }
#endif

#include <Eigen/Sparse>

#define DecVarsSize ((Tnx * Tph) + (Tnu * Tch) + 1)
#define StateEqSize (Tph * Tnx)
#define StateIneqSize (2 * Tph * Tny)

#define dbg(x) Logger(x)

namespace mpc {

int Logger::verbose = 0;
Logger::level Logger::logLevel = Logger::level::INFO;

template <std::size_t M, std::size_t N>
using mat = Eigen::Matrix<double, M, N>;

template <std::size_t M, std::size_t N, std::size_t K>
using mat3 = std::array<Eigen::Matrix<double, M, N>, K>;

template <std::size_t N>
using cvec = Eigen::Matrix<double, N, 1>;

template <std::size_t N>
using rvec = Eigen::Matrix<double, 1, N>;

template <std::size_t Tph, std::size_t Tnx, std::size_t Tnu>
using ObjFunHandle = std::function<double(mat<Tph + 1, Tnx>, mat<Tph + 1, Tnu>, double)>;
template <std::size_t Tcon, std::size_t Tph, std::size_t Tnx, std::size_t Tnu>
using IConFunHandle = std::function<cvec<Tcon>(mat<Tph + 1, Tnx>, mat<Tph + 1, Tnu>, double)>;
template <std::size_t Tcon, std::size_t Tph, std::size_t Tnx, std::size_t Tnu>
using EConFunHandle = std::function<cvec<Tcon>(mat<Tph + 1, Tnx>, mat<Tph + 1, Tnu>)>;

template <std::size_t Tnx, std::size_t Tnu>
using StateFunHandle = std::function<cvec<Tnx>(cvec<Tnx>, cvec<Tnu>)>;
using OutFunHandle = std::function<void(void)>;

template <std::size_t Tnu>
struct Result {
    Result() = default;

    int retcode;
    double cost;
    cvec<Tnu> cmd;
};

enum constraints_type {
    INEQ,
    EQ,
    UINEQ,
    UEQ
};
} // namespace mpc