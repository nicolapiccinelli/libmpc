#pragma once

#include <mpc/IComponent.hpp>
#include <mpc/Mapping.hpp>
#include <mpc/Types.hpp>

namespace mpc
{

    /**
     * @brief Abstract base class for the linear and non-linear mpc
     * interfaces
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
            ts = 0;
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
         * @brief Set the mapping object reference
         *
         * @param m the mapping object
         */
        void setMapping(Mapping<sizer> &m)
        {
            mapping = m;
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

        cvec<sizer.nx> x0;
        mat<sizer.ph + 1, sizer.nx> Xmat;
        mat<sizer.ph + 1, sizer.nu> Umat;

        double e;
        double ts;
    };

} // namespace mpc
