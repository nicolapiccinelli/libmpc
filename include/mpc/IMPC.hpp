/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IDimensionable.hpp>
#include <mpc/IOptimizer.hpp>
#include <mpc/Profiler.hpp>

namespace mpc
{
    /**
     * @brief Abstract class defining the shared API between linear
     * and non-linear MPC
     *
     * @tparam Tnx dimension of the state space
     * @tparam Tnu dimension of the input space
     * @tparam Tndu dimension of the measured disturbance space
     * @tparam Tny dimension of the output space
     * @tparam Tph length of the prediction horizon
     * @tparam Tch length of the control horizon
     * @tparam Tineq number of the user inequality constraints
     * @tparam Teq number of the user equality constraints
     */
    template <MPCSize sizer>
    class IMPC : public IDimensionable<sizer>
    {
    protected:
        using IDimensionable<sizer>::nu;
        using IDimensionable<sizer>::nx;
        using IDimensionable<sizer>::ndu;
        using IDimensionable<sizer>::ny;
        using IDimensionable<sizer>::ph;
        using IDimensionable<sizer>::ch;
        using IDimensionable<sizer>::ineq;
        using IDimensionable<sizer>::eq;

    public:
        /**
         * @brief Set the discretization time step to use for numerical integration
         *
         * @return true
         * @return false
         */
        virtual bool setContinuosTimeModel(const double) = 0;
        /**
         * @brief Set the scaling factor for the control input. This can be used to normalize
         * the control input with respect to the different measurment units
         */
        virtual void setInputScale(const cvec<sizer.nu>) = 0;
        /**
         * @brief Set the scaling factor for the dynamical system's states variables.
         * This can be used to normalize the dynamical system's states variables
         * with respect to the different measurment units
         */
        virtual void setStateScale(const cvec<sizer.nx>) = 0;
        /**
         * @brief Set the solver specific parameters
         */
        virtual void setOptimizerParameters(const Parameters &) = 0;

        /**
         * @brief Implements the initilization hook to provide shared initilization logic
         * and forwards the hook through the setup hook for linear and non-linear interface
         * specific initilization
         */
        void onInit()
        {
            profiler.reset();
            onSetup();
        };

        /**
         * @brief Set the logger level
         *
         * @param l logger level desired
         * @return true
         * @return false
         */
        bool setLoggerLevel(Logger::log_level l)
        {
            Logger::instance().setLevel(l);
            return true;
        }

        /**
         * @brief Set the prefix used on any log message
         *
         * @param prefix prefix desired
         * @return true
         * @return false
         */
        bool setLoggerPrefix(std::string prefix)
        {
            Logger::instance().setPrefix(prefix);
            return true;
        }

        /**
         * @brief Compute the optimal control action
         *
         * @param x0 system's variables initial condition
         * @param lastU last optimal control action
         * @return Result<Tnu> optimization result
         */
        Result<sizer.nu> step(const cvec<sizer.nx> x0, const cvec<sizer.nu> lastU)
        {
            onModelUpdate(x0);

            profiler.solutionStart();
            optPtr->run(x0, lastU);
            auto duration_s = profiler.solutionEnd<sizer.nu>(optPtr->result);

            Logger::instance().log(Logger::log_type::INFO)
                << "Optimization step completed" << std::endl
                << "duration: " << duration_s.count() << " (sec)" << std::endl
                << "status: " << optPtr->result.status << " (opt code: " << optPtr->result.retcode << ")" << std::endl
                << "cost: " << optPtr->result.cost << std::endl;

            return optPtr->result;
        }

        /**
         * @brief Get the last optimal control action
         *
         * @return Result<Tnu> last optimal control action
         */
        Result<sizer.nu> getLastResult()
        {
            return optPtr->result;
        }

        /**
         * @brief Get the Optimal Sequence object
         *
         * @return OptSequence<sizer.nx, sizer.ny, sizer.nu, sizer.ph> last optimal sequence (zeros if optimization fails)
         */
        OptSequence<sizer.nx, sizer.ny, sizer.nu, sizer.ph> getOptimalSequence()
        {
            return optPtr->sequence;
        }
        
        /**
         * @brief Returns the execution statistics of the profiler.
         *
         * The `getExecutionStats` function retrieves the execution statistics of the profiler, 
         * which includes the minimum, maximum, average, and standard deviation of the solution time, as well as the number of solutions and the total time.
         * This function returns a `SolutionStats` struct that contains all of these statistics.
         *
         * @return A `SolutionStats` struct that contains the minimum, maximum, average, and standard deviation of the solution time, as well as the number of solutions and the total time.
         *
         * @see addSolutionTime()
         */
        const SolutionStats &getExecutionStats()
        {
            return profiler.getStats();
        }

        /**
         * @brief Resets the profiler statistics to their default values.
         *
         * This function resets the profiler statistics to their default values, clearing out any previous values that were
         * recorded.
         *
         * @param None
         *
         * @return None
         */
        void resetStats()
        {
            profiler.reset();
        }

    protected:
        /**
         * @brief Initilization hook for the linear and non-linear interfaces
         */
        virtual void onSetup() = 0;
        /**
         * @brief Dynamical system initial condition update hook
         */
        virtual void onModelUpdate(const cvec<sizer.nx>) = 0;

        IOptimizer<sizer> *optPtr;
        Profiler profiler;
    };
}