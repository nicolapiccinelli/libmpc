/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/NLMPC/Base.hpp>

namespace mpc
{

    /**
     * @brief Managment of the objective function for the non-linear mpc
     *
     * @tparam sizer.nx dimension of the state space
     * @tparam sizer.nu dimension of the input space
     * @tparam Tny dimension of the output space
     * @tparam Tph length of the prediction horizon
     * @tparam Tch length of the control horizon
     * @tparam Tineq number of the user inequality constraints
     * @tparam Teq number of the user equality constraints
     */
    template <MPCSize sizer>
    class Objective : public Base<sizer>
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
        /**
         * @brief Internal structure containing the value and the gradient
         * of the evaluated constraints.
         */
        struct Cost
        {
            double value;
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> grad;
        };

        Objective() = default;
        ~Objective() = default;

        /**
         * @brief Initialization hook override used to perform the
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed.
         */
        void onInit()
        {
            x0.resize(nx());
            Xmat.resize((ph() + 1), nx());
            Umat.resize((ph() + 1), nu());
            Jx.resize(nx(), ph());
            Jmv.resize(nu(), ph());

            Je = 0;
        }

        /**
         * @brief Set the objective function to be minimized
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setObjective(
            const typename Base<sizer>::ObjFunHandle handle)
        {
            checkOrQuit();
            return fuser = handle, true;
        }

        /**
         * @brief Evaluate the objective function at the desired optimal vector
         *
         * @param x internal optimal vector
         * @param hasGradient request the computation of the gradient
         * @return Cost associated cost
         */
        Cost evaluate(
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            bool hasGradient)
        {
            checkOrQuit();

            Cost c;
            c.grad.resize(((ph() * nx()) + (nu() * ch()) + 1));

            mapping.unwrapVector(x, x0, Xmat, Umat, e);
            c.value = fuser(Xmat, model.getOutput(Xmat, Umat), Umat, e);

            if (hasGradient)
            {
                computeJacobian(Xmat, Umat, c.value, e);

                for (int i = 0; i < Xmat.cols(); i++)
                {
                    Xmat.col(i) /= 1.0 / mapping.StateScaling()(i);
                }

                int counter = 0;

                // #pragma omp parallel for
                for (int j = 0; j < (int)Jx.cols(); j++)
                {
                    for (int i = 0; i < (int)Jx.rows(); i++)
                    {
                        c.grad(counter++) = Jx(i, j);
                    }
                }

                cvec<(sizer.ph * sizer.nu)> JmvVectorized;
                JmvVectorized.resize((ph() * nu()));

                int vec_counter = 0;
                // #pragma omp parallel for
                for (int j = 0; j < (int)Jmv.cols(); j++)
                {
                    for (int i = 0; i < (int)Jmv.rows(); i++)
                    {
                        JmvVectorized(vec_counter++) = Jmv(i, j);
                    }
                }

                cvec<(sizer.nu * sizer.ch)> res;
                res = mapping.Iz2u().transpose() * JmvVectorized;
                // #pragma omp parallel for
                for (size_t j = 0; j < (ch() * nu()); j++)
                {
                    c.grad(counter++) = res(j);
                }

                c.grad(((ph() * nx()) + (nu() * ch()) + 1) - 1) = Je;
            }

            Logger::instance().log(Logger::log_type::DETAIL)
                << "("
                << niteration
                << ") Objective function value: \n"
                << std::setprecision(10)
                << c.value
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "("
                    << niteration
                    << ") Gradient not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "("
                    << niteration
                    << ") Objective function gradient: \n"
                    << std::setprecision(10)
                    << c.grad
                    << std::endl;
            }

            // debug information
            niteration++;

            return c;
        }

    private:
        /**
         * @brief Approximate the objective function Jacobian matrices
         *
         * @param x0 current state configuration
         * @param u0 current optimal input configuration
         * @param f0 current user inequality constraints values
         * @param e0 current slack value
         */
        void computeJacobian(
            mat<(sizer.ph + 1), sizer.nx> x0,
            mat<(sizer.ph + 1), sizer.nu> u0,
            double f0,
            double e0)
        {
            double dv = 1e-6;

            Jx.setZero();
            // TODO support measured disturbaces
            Jmv.setZero();

            mat<(sizer.ph + 1), sizer.nx> Xa;
            Xa = x0.cwiseAbs().unaryExpr([](double d)
                                         { return (d < 1) ? 1 : d; });

            // #pragma omp parallel for
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    int ix = i + 1;
                    double dx = dv * Xa.array()(j);
                    x0(ix, j) = x0(ix, j) + dx;
                    double f = fuser(x0, model.getOutput(x0, u0), u0, e0);
                    x0(ix, j) = x0(ix, j) - dx;
                    double df = (f - f0) / dx;
                    Jx(j, i) = df;
                }
            }

            mat<(sizer.ph + 1), sizer.nu> Ua;
            Ua = u0.cwiseAbs().unaryExpr([](double d)
                                         { return (d < 1) ? 1 : d; });

            // #pragma omp parallel for
            for (size_t i = 0; i < (ph() - 1); i++)
            {
                // TODO support measured disturbaces
                for (size_t j = 0; j < nu(); j++)
                {
                    int k = j;
                    double du = dv * Ua.array()(k);
                    u0(i, k) = u0(i, k) + du;
                    double f = fuser(x0, model.getOutput(x0, u0), u0, e0);
                    u0(i, k) = u0(i, k) - du;
                    double df = (f - f0) / du;
                    Jmv(j, i) = df;
                }
            }

            // TODO support measured disturbaces
            // #pragma omp parallel for
            for (size_t j = 0; j < nu(); j++)
            {
                int k = j;
                double du = dv * Ua.array()(k);
                u0((ph() - 1), k) = u0((ph() - 1), k) + du;
                u0(ph(), k) = u0(ph(), k) + du;
                double f = fuser(x0, model.getOutput(x0, u0), u0, e0);
                u0((ph() - 1), k) = u0((ph() - 1), k) - du;
                u0(ph(), k) = u0(ph(), k) - du;
                double df = (f - f0) / du;
                Jmv(j, (ph() - 1)) = df;
            }

            double ea = fmax(1e-6, abs(e0));
            double de = ea * dv;
            double f1 = fuser(x0, model.getOutput(x0, u0), u0, e0 + de);
            double f2 = fuser(x0, model.getOutput(x0, u0), u0, e0 - de);
            Je = (f1 - f2) / (2 * de);
        }

        typename Base<sizer>::ObjFunHandle fuser = nullptr;

        mat<sizer.nx, sizer.ph> Jx;
        mat<sizer.nu, sizer.ph> Jmv;

        double Je;

        using Base<sizer>::mapping;
        using Base<sizer>::model;
        using Base<sizer>::x0;
        using Base<sizer>::Xmat;
        using Base<sizer>::Umat;
        using Base<sizer>::e;
        using Base<sizer>::niteration;
    };
} // namespace mpc
