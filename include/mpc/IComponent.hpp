/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/Dim.hpp>
#include <mpc/Types.hpp>

namespace mpc
{

    /**
     * @brief Abstract class for the explicit initializable classes
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
    class IComponent : public IDimensionable<sizer>
    {
    public:
        IComponent()
        {
            isInitialized = false;
        }

        /**
         * @brief Provide the explicit initialization of the component
         *
         * @param nx dimension of the state space
         * @param nu dimension of the input space
         * @param ndu dimension of the measured disturbance space
         * @param ny dimension of the output space
         * @param ph length of the prediction horizon
         * @param ch length of the control horizon
         * @param ineq number of the user inequality constraints
         * @param eq number of the user equality constraints
         */
        void initialize(
            int nx = sizer.nx, int nu = sizer.nu, int ndu = sizer.ndu, int ny = sizer.ny,
            int ph = sizer.ph, int ch = sizer.ch, int ineq = sizer.ineq, int eq = sizer.eq)
        {
            isInitialized = true;
            setDimension(nx, nu, ndu, ny, ph, ch, ineq, eq);
        }

    protected:
        // this is used just to avoid explicit construction of this class
        virtual ~IComponent() = default;

        /**
         * @brief Initialization hook used to perform sub-classes
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        virtual void onInit() = 0;

        /**
         * @brief Check if the object has been correctly initialized. In case
         * the initialization has not been performed yet, the library exits
         * causing a crash
         */
        inline void checkOrQuit()
        {
            if (!isInitialized)
            {
                Logger::instance().log(Logger::log_type::ERROR) << "MPC library is not initialized, quitting..." << std::endl;
                exit(-1);
            }
        }

    private:
        using IDimensionable<sizer>::setDimension;
        bool isInitialized;
    };

} // namespace mpc
