/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IComponent.hpp>

namespace mpc
{
    /**
     * @brief Managment of the user-defined and sytem dynamic constraints
     * for the non-linear mpc
     *
     * @tparam Tnx dimension of the state space
     * @tparam Tnu dimension of the input space
     * @tparam Tny dimension of the output space
     * @tparam Tph length of the prediction horizon
     * @tparam Tch length of the control horizon
     * @tparam Tineq number of the user inequality constraints
     * @tparam Teq number of the user equality constraints
     */
    template <MPCSize sizer>
    class Model : public IComponent<sizer>
    {
    private:
        using IComponent<sizer>::checkOrQuit;
        using IDimensionable<sizer>::nu;
        using IDimensionable<sizer>::nx;
        using IDimensionable<sizer>::ndu;
        using IDimensionable<sizer>::ny;
        using IDimensionable<sizer>::ph;
        using IDimensionable<sizer>::ch;
        using IDimensionable<sizer>::ineq;
        using IDimensionable<sizer>::eq;

    public:
        Model() : IComponent<sizer>()
        {
            isContinuosTime = false;
        }

        /**
         * @brief Initialization hook override used to perform the
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed.
         */
        void onInit()
        {
        }

        /**
         * @brief Return if the dynamical system has an output function
         *
         * @return true
         * @return false
         */
        bool hasOutputModel()
        {
            checkOrQuit();
            return outUser != nullptr;
        }

        /**
         * @brief Get the output vector from the model
         *
         * @param Xmat the state vectors sequence along the prediction horizon
         * @param Umat the input vectors sequence along the prediction horizon
         * @return mat<sizer.ph + 1, sizer.ny> the output vector sequence along the prediction horizon
         */
        mat<sizer.ph + 1, sizer.ny> getOutput(
            const mat<(sizer.ph + 1), sizer.nx>& Xmat,
            const mat<(sizer.ph + 1), sizer.nu>& Umat)
        {
            checkOrQuit();

            mat<sizer.ph + 1, sizer.ny> Ymat;
            Ymat.resize(ph() + 1, ny());
            Ymat.setZero();

            if (hasOutputModel())
            {
                for (size_t i = 0; i < ph() + 1; i++)
                {
                    cvec<sizer.ny> YmatRow;
                    YmatRow.resize(ny());
                    YmatRow.setZero();

                    outUser(YmatRow, Xmat.row(i), Umat.row(i), i);
                    Ymat.row(i) = YmatRow;
                }
            }

            return Ymat;
        }

        /**
         * @brief Set if the provided dynamical model is in continuos time
         *
         * @param isContinuous system dynamics is defined in countinuos time
         * @param Ts discretization sample time, in general this is the inverse of the control loop frequency
         * @return true
         * @return false
         */
        bool setContinuos(bool isContinuous, double Ts = 0)
        {
            sampleTime = Ts;
            isContinuosTime = isContinuous;
            return true;
        }

        /**
         * @brief Set the system's states update function (e.g. the vector field)
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setStateModel(
            const typename IDimensionable<sizer>::StateFunHandle handle)
        {
            checkOrQuit();
            return vectorField = handle, true;
        }

        /**
         * @brief Set the system's output function (e.g. the state/output mapping)
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setOutputModel(
            const typename IDimensionable<sizer>::OutFunHandle handle)
        {
            checkOrQuit();
            return outUser = handle, true;
        }

        bool isContinuosTime;
        double sampleTime;

        typename IDimensionable<sizer>::StateFunHandle vectorField = nullptr;

    private:
        typename IDimensionable<sizer>::OutFunHandle outUser = nullptr;
    };
} // namespace mpc
