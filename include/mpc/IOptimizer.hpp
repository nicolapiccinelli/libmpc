/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IComponent.hpp>

namespace mpc
{
    /**
     * @brief Abstract class defining the optimizer interface for
     * linear and non-linear MPC
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
    class IOptimizer : public IComponent<sizer>
    {
    public:
        virtual ~IOptimizer() {}

        /**
         * @brief Initialization hook override used to perform the optimizer interfaces
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        virtual void onInit() = 0;
        /**
         * @brief Abstract setter for the optimizer parameters
         *
         * @param param desired parameters (refers to the optimizer specific for the meaning of the parameters)
         */
        virtual void setParameters(const Parameters &param) = 0;
        /**
         * @brief Abstract caller to perform the optimization
         *
         * @param x0 system's variables initial condition
         * @param u0 control action initial condition for warm start
         * @return Result<Tnu> optimization result
         */
        virtual void run(const cvec<sizer.nx> &x0, const cvec<sizer.nu> &u0) = 0;

        Result<sizer.nu> result;
        OptSequence<sizer.nx, sizer.ny, sizer.nu, sizer.ph> sequence;

    protected:
        double currentSlack;
        bool hard;
    };
}