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
     * @brief Abstract class for all the classes which need access
     * to the problem dimensions and to the function handlers types
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
    class IDimensionable
    {
    public:
        IDimensionable()
        {
        }

    protected:
        // this is used just to avoid explicit construction of this class
        virtual ~IDimensionable() = default;

        /**
         * @brief Initialize the dimensions of the optimization problem
         * and then invokes the onInit method to perform extra initialization.
         * In case of static allocation the dimensions are inferred from the
         * template class parameters
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
        void setDimension(
            int nx = sizer.nx, int nu = sizer.nu, int ndu = sizer.ndu, int ny = sizer.ny,
            int ph = sizer.ph, int ch = sizer.ch, int ineq = sizer.ineq, int eq = sizer.eq)
        {
            assert(nx >= 0 && nu >= 0 && ndu >= 0 && ny >= 0 && ph > 0 && ch > 0 && ineq >= 0 && eq >= 0);

            runtime_size_nx = nx;
            runtime_size_nu = nu;
            runtime_size_ndu = ndu;
            runtime_size_ny = ny;
            runtime_size_ph = ph;
            runtime_size_ch = ch;
            runtime_size_ineq = ineq;
            runtime_size_eq = eq;

            onInit();
        }

        size_t nx() { return runtime_size_nx; }
        size_t nu() { return runtime_size_nu; }
        size_t ndu() { return runtime_size_ndu; }
        size_t ny() { return runtime_size_ny; }
        size_t ph() { return runtime_size_ph; }
        size_t ch() { return runtime_size_ch; }
        size_t ineq() { return runtime_size_ineq; }
        size_t eq() { return runtime_size_eq; }

        /**
         * @brief Initialization hook used to perform sub-classes
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        virtual void onInit() = 0;

        /**
         * @brief User-defined function handle for the non-linear MPC
         * objective function. The arguments of the function are the
         * state, output and input vectors along the horizon while the last
         * term is the slack variable. The function must return the 
         * scalar value of the objective function
         */
        using ObjFunHandle = std::function<double(
            const mat<sizer.ph + 1, sizer.nx> &,
            const mat<sizer.ph + 1, sizer.ny> &,
            const mat<sizer.ph + 1, sizer.nu> &,
            const double &)>;

        /**
         * @brief User-defined function handle for the non-linear MPC
         * inequality constraints function. The arguments of the function 
         * are the inequality vector containg the value of each term and the
         * current state, output and input vectors along the horizon. The last
         * term is the slack variable
         */
        using IConFunHandle = std::function<void(
            cvec<sizer.ineq> &,
            const mat<sizer.ph + 1, sizer.nx> &,
            const mat<sizer.ph + 1, sizer.ny> &,
            const mat<sizer.ph + 1, sizer.nu> &,
            const double &)>;

        /**
         * @brief User-defined function handle for the non-linear MPC
         * equality constraints function. The arguments of the function
         * are the equality vector containg the value of each term and the
         * current state and input vectors along the horizon
         */
        using EConFunHandle = std::function<void(
            cvec<sizer.eq> &,
            const mat<sizer.ph + 1, sizer.nx> &,
            const mat<sizer.ph + 1, sizer.nu> &)>;

        /**
         * @brief User-defined function handle for the non-linear MPC
         * dynamical system model. The arguments of the function are the
         * vector field (or next state) and the current state and input
         * vectors along the horizon. The last argument is the step of 
         * the horizon on which the system output is evaluated
         */
        using StateFunHandle = std::function<void(
            cvec<sizer.nx> &,
            const cvec<sizer.nx> &,
            const cvec<sizer.nu> &,
            const unsigned int &)>;

        /**
         * @brief User-defined function handle for the non-linear MPC
         * dynamical system output model. The arguments of the function are
         * the system output and the current state and input vectors along
         * the horizon. The last argument is the step of the horizon on which
         * the system output is evaluated
         */
        using OutFunHandle = std::function<void(
            cvec<sizer.ny> &,
            const cvec<sizer.nx> &,
            const cvec<sizer.nu> &,
            const unsigned int &)>;

    private:
        size_t runtime_size_nx;
        size_t runtime_size_nu;
        size_t runtime_size_ndu;
        size_t runtime_size_ny;
        size_t runtime_size_ph;
        size_t runtime_size_ch;
        size_t runtime_size_ineq;
        size_t runtime_size_eq;
    };

} // namespace mpc
