/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IComponent.hpp>
#include <mpc/NLMPC/Mapping.hpp>
#include <mpc/NLMPC/Model.hpp>
#include <mpc/Types.hpp>

namespace mpc
{

    /**
     * @brief Abstract base class for non-linear mpc components
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
    class Base : public IComponent<sizer>
    {

    public:
        Base() : IComponent<sizer>()
        {
            e = 0;
            niteration = 0;
        }

        /**
         * @brief Initialization hook override used to perform mpc interfaces
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        void onInit() = 0;

        /**
         * @brief Set the model and the mapping object references
         *
         * @param sysModel the model object
         * @param map the mapping object
         */
        void setModel(Model<sizer> &sysModel, Mapping<sizer> &map)
        {
            mapping = map;
            model = sysModel;
        }

        /**
         * @brief Set the current state of the optimizer
         *
         * @param currState
         */
        void setCurrentState(const cvec<sizer.nx> currState)
        {
            x0 = currState;
            niteration = 1;
        }

        // debug information
        int niteration;

    protected:
        Mapping<sizer> mapping;
        Model<sizer> model;

        cvec<sizer.nx> x0;
        mat<sizer.ph + 1, sizer.nx> Xmat;
        mat<sizer.ph + 1, sizer.nu> Umat;

        double e;
    };

} // namespace mpc
