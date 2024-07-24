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
     * @brief Horizon slice to indicate a subset of the horizon
     */
    struct HorizonSlice
    {
        int start;
        int end;

        /**
         * @brief Construct a new horizon slice
         * 
         * @param start the starting index of the slice (zero-based)
         * @param end the ending index of the slice (zero-based)
         */
        HorizonSlice(int start, int end) : start(start), end(end) 
        {
            
        }

        /**
         * @brief Create a slice to indicate the whole horizon 
         * 
         * @return HorizonSlice the instance of the slice representing the whole horizon
         */
        static HorizonSlice all()
        {
            return HorizonSlice{-1, -1};
        }
    };

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
        /// @brief Enable the warm start of the optimization (enabling the warm start
        // can speed up the optimization process if the optimization variables
        // are close to the optimal solution)
        bool enable_warm_start = false;
    };

    /**
     * @brief Non-linear optimizer parameters
     * (SEE NLOPT DOCUMENTATION FOR MORE DETAILS)
     */
    struct NLParameters : Parameters
    {
        NLParameters() = default;

        /// @brief the percentage of the objective function value below which the optimization is considered converged
        // negative value means that the convergence check is disabled
        double relative_ftol = -1;
        /// @brief the percentage of the optimization variables below which the optimization is considered converged
        // negative value means that the convergence check is disabled
        double relative_xtol = -1;
        /// @brief the absolute value of the objective function value below which the optimization is considered converged
        // negative value means that the convergence check is disabled
        double absolute_ftol = -1;
        /// @brief the absolute value of the optimization variables below which the optimization is considered converged
        // negative value means that the convergence check is disabled
        double absolute_xtol = -1;

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
        Result() : solver_status(0), cost(0), status(ResultStatus::UNKNOWN), solver_status_msg(""), is_feasible(false)
        {
            cmd.setZero();
        }

        int solver_status;
        bool is_feasible;
        std::string solver_status_msg;
        double cost;
        ResultStatus status;
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

// Define a macro to conditionally resize Eigen vectors or matrices
// TODO: the condition to check if the size is dynamic should be done
// on the whole sizer not just on the nx value (even if this condition
// is enough for the current implementation, it is better to be more
// general)
#define COND_RESIZE_MAT(sizer, matrix, desired_rows, desired_cols) \
    if constexpr (sizer.nx.value == Eigen::Dynamic)                \
    {                                                              \
        matrix.resize(desired_rows, desired_cols);                 \
    }                                                              \
    else                                                           \
    {                                                              \
    }

// Define a macro to conditionally resize Eigen vectors and resort to
// the previous macro by setting the desired_cols to 1
#define COND_RESIZE_CVEC(sizer, vector, desired_size) \
    COND_RESIZE_MAT(sizer, vector, desired_size, 1)

// Define a macro to conditionally resize Eigen vectors and resort to
// the previous macro by setting the desired_rows to 1
#define COND_RESIZE_RVEC(sizer, vector, desired_size) \
    COND_RESIZE_MAT(sizer, vector, 1, desired_size)

} // namespace mpc
