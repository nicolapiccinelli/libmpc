/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpc/Logger.hpp>
#include <vector>
#include <chrono>

#if SHOW_STACKTRACE == 1

#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED
#include <boost/stacktrace.hpp>

#undef eigen_assert
#define eigen_assert(x)                              \
    if (!(x))                                        \
    {                                                \
        std::cout << boost::stacktrace::stacktrace() \
                  << std::endl                       \
                  << std::endl;                      \
        exit(-1);                                    \
    }
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace mpc
{
    template <
        int M = Eigen::Dynamic,
        int N = Eigen::Dynamic>
    using mat = Eigen::Matrix<double, M, N>;

    using smat = Eigen::SparseMatrix<double, Eigen::ColMajor>;

    template <
        int N = Eigen::Dynamic>
    using cvec = Eigen::Matrix<double, N, 1>;

    template <
        int N = Eigen::Dynamic>
    using rvec = Eigen::Matrix<double, 1, N>;

    /**
     * @brief Optimization result status
     */
    enum ResultStatus
    {
        SUCCESS,
        MAX_ITERATION,
        INFEASIBLE,
        ERROR,
        UNKNOWN
    };

    /**
     * @brief Shared optimizer parameters
     */
    struct Parameters
    {
    protected:
        Parameters() = default;
        virtual ~Parameters() = default;

    public:
        /// @brief Set the maximum number of iterations before stopping the optimization
        int maximum_iteration = 100;
        /// @brief Set the maximum time before stopping the optimization (in seconds)
        double time_limit = 0;
    };

    /**
     * @brief Non-linear optimizer parameters 
     * (SEE NLOPT DOCUMENTATION FOR MORE DETAILS)
     */
    struct NLParameters : Parameters
    {
        NLParameters() = default;

        /// @brief the percentage of the objective function value below which the optimization is considered converged
        double relative_ftol = 1e-10;
        /// @brief the percentage of the optimization variables below which the optimization is considered converged
        double relative_xtol = 1e-10;
        /// @brief the absolute value of the objective function value below which the optimization is considered converged
        double absolute_ftol = 1e-10;
        /// @brief the absolute value of the optimization variables below which the optimization is considered converged
        double absolute_xtol = 1e-10;

        /// @brief If enabled, the slack variable is constrained to be zero (forcing the inequality constraints to be hard constraints)
        bool hard_constraints = true;
    };

    /**
     * @brief Linear optimizer parameters 
     * (SEE OSQP DOCUMENTATION FOR MORE DETAILS)
     */
    struct LParameters : Parameters
    {
        LParameters() = default;

        double alpha = 1.6;
        double rho = 1e-6;

        double eps_rel = 1e-4;
        double eps_abs = 1e-4;
        double eps_prim_inf = 1e-3;
        double eps_dual_inf = 1e-3;
        
        bool enable_warm_start = false;

        bool verbose = false;
        bool adaptive_rho = true;
        bool polish = true;
    };

    /**
     * @brief Optimization control input result
     *
     * @tparam Tnu dimension of the input space
     */
    template <int Tnu = Eigen::Dynamic>
    struct Result
    {
        Result() : retcode(0), cost(0), status(ResultStatus::UNKNOWN)
        {
            cmd.setZero();
        }

        int retcode;
        double cost;
        ResultStatus status;
        std::string status_msg;
        cvec<Tnu> cmd;
    };

    template <
        int Tnx = Eigen::Dynamic,
        int Tny = Eigen::Dynamic,
        int Tnu = Eigen::Dynamic,
        int Tph = Eigen::Dynamic>
    struct OptSequence
    {
        OptSequence()
        {
        }

        mat<Tph, Tnx> state;
        mat<Tph, Tny> output;
        mat<Tph, Tnu> input;
    };

    enum constraints_type
    {
        INEQ,
        EQ,
        UINEQ,
        UEQ
    };

    /**
     * @brief Utility to get the dimension based on the input flag
     *
     * @param n value
     * @param c flag
     * @return constexpr int input dimension or -1
     */
    inline constexpr int make_dimension(const int n, bool c)
    {
        if (c)
        {
            return n;
        }
        else
        {
            return Eigen::Dynamic;
        }
    }

    constexpr double inf = std::numeric_limits<double>::infinity();

} // namespace mpc
