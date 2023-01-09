/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IComponent.hpp>

namespace mpc
{

    /**
     * @brief Utility class for manipulating the scaling and the transformations
     * necessary to compute the non-linear optimization problem
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
    class Mapping : public IComponent<sizer>
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
        Mapping() : IComponent<sizer>()
        {
        }

        /**
         * @brief Initialization hook override. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        void onInit()
        {
            Iz2uMat.resize((ph() * nu()), (nu() * ch()));
            Iu2zMat.resize((nu() * ch()), (ph() * nu()));
            Sz2uMat.resize(nu(), nu());
            Su2zMat.resize(nu(), nu());

            input_scaling.resize(nu());
            state_scaling.resize(nx());
            inverse_state_scaling.resize(nx());

            input_scaling.setOnes();
            state_scaling.setOnes();
            inverse_state_scaling.setOnes();

            computeMapping();
        }

        /**
         * @brief Set the input scaling matrix
         *
         * @param scaling input scaling vector
         */
        void setInputScaling(const cvec<sizer.nu> scaling)
        {
            input_scaling = scaling;
            computeMapping();
        }

        /**
         * @brief Set the state scaling matrix
         *
         * @param scaling state scaling vector
         */
        void setStateScaling(const cvec<sizer.nx> scaling)
        {
            state_scaling = scaling;
            inverse_state_scaling = scaling.cwiseInverse();
        }

        /**
         * @brief Accesor to the optimal vector to input mapping matrix
         *
         * @return mat<(sizer.ph * sizer.nu), (sizer.nu * sizer.ch)> mapping matrix
         */
        mat<(sizer.ph * sizer.nu), (sizer.nu * sizer.ch)> Iz2u()
        {
            checkOrQuit();
            return Iz2uMat;
        }

        /**
         * @brief Accesor to the inverse of the optimal vector to input mapping matrix
         *
         * @return mat<(sizer.nu * sizer.ch), (sizer.ph * sizer.nu)> mapping matrix
         */
        mat<(sizer.nu * sizer.ch), (sizer.ph * sizer.nu)> Iu2z()
        {
            checkOrQuit();
            return Iu2zMat;
        }

        /**
         * @brief Accesor to the scaled optimal vector to input mapping matrix
         *
         * @return mat<sizer.nu, sizer.nu> mapping matrix
         */
        mat<sizer.nu, sizer.nu> Sz2u()
        {
            checkOrQuit();
            return Sz2uMat;
        }

        /**
         * @brief Accesor to the inverse of scaled optimal vector to input mapping matrix
         *
         * @return mat<sizer.nu, sizer.nu> mapping matrix
         */
        mat<sizer.nu, sizer.nu> Su2z()
        {
            checkOrQuit();
            return Su2zMat;
        }

        /**
         * @brief Get the current state scaling vector
         *
         * @return cvec<Tnx> scaling vector
         */
        cvec<sizer.nx> StateScaling()
        {
            checkOrQuit();
            return state_scaling;
        }

        /**
         * @brief Get the inverse of the current state scaling vector
         *
         * @return cvec<Tnx> scaling vector
         */
        cvec<sizer.nx> StateInverseScaling()
        {
            checkOrQuit();
            return inverse_state_scaling;
        }

        /**
         * @brief Get the current input scaling vector
         *
         * @return cvec<Tnx> scaling vector
         */
        cvec<sizer.nu> InputScaling()
        {
            checkOrQuit();
            return input_scaling;
        }

        /**
         * @brief Convert from optimal vector to the state, input and slackness sub-vectors
         *
         * @param x current optimal vector
         * @param x0 current system's dynamics initial condition
         * @param Xmat state vector along the prediction horizon
         * @param Umat input vector along the prediction horizon
         * @param slack slackness values along the prediction horizon
         */
        void unwrapVector(
            const cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            const cvec<sizer.nx> x0,
            mat<(sizer.ph + 1), sizer.nx> &Xmat,
            mat<(sizer.ph + 1), sizer.nu> &Umat,
            double &slack)
        {
            checkOrQuit();

            cvec<(sizer.nu * sizer.ch)> u_vec;
            u_vec = x.middleRows((ph() * nx()), (nu() * ch()));

            mat<sizer.ph + 1, sizer.nu> Umv;
            Umv.resize((ph() + 1), nu());

            cvec<(sizer.ph * sizer.nu)> tmp_mult;
            tmp_mult = Iz2uMat * u_vec;
            mat<sizer.nu, sizer.ph> tmp_mapped;
            tmp_mapped = Eigen::Map<mat<sizer.nu, sizer.ph>>(tmp_mult.data(), nu(), ph());

            Umv.setZero();
            Umv.middleRows(0, ph()) = tmp_mapped.transpose();
            Umv.row(ph()) = Umv.row(ph() - 1);

            Xmat.setZero();
            Xmat.row(0) = x0.transpose();
            for (size_t i = 1; i < (ph() + 1); i++)
            {
                Xmat.row(i) = x.middleRows(((i - 1) * nx()), nx()).transpose();
            }

            for (int i = 0; i < Xmat.cols(); i++)
            {
                Xmat.col(i) /= 1.0 / state_scaling(i);
            }

            // TODO add disturbaces manipulated vars
            Umat.setZero();
            Umat.block(0, 0, ph() + 1, nu()) = Umv;

            slack = x(x.size() - 1);
        }

    protected:
        cvec<sizer.nu> input_scaling;
        cvec<sizer.nx> state_scaling, inverse_state_scaling;

    private:
        /**
         * @brief Utility function to compute the mapping matrices
         */
        void computeMapping()
        {
            static cvec<sizer.ch> m;
            m.resize(ch());

            for (size_t i = 0; i < ch(); i++)
            {
                m(i) = 1;
            }

            m(ch() - 1) = ph() - ch() + 1;

            Iz2uMat.setZero();
            Iu2zMat = Iz2uMat.transpose();

            Sz2uMat.setZero();
            Su2zMat.setZero();
            for (int i = 0; i < Sz2uMat.rows(); ++i)
            {
                Sz2uMat(i, i) = input_scaling(i);
                Su2zMat(i, i) = 1.0 / input_scaling(i);
            }

            // TODO implement linear interpolation
            int ix = 0;
            int jx = 0;
            for (size_t i = 0; i < ch(); i++)
            {
                Iu2zMat.block(ix, jx, nu(), nu()) = Su2zMat;
                for (int j = 0; j < m[i]; j++)
                {
                    Iz2uMat.block(jx, ix, nu(), nu()) = Sz2uMat;
                    jx += nu();
                }
                ix += nu();
            }
        }

        mat<(sizer.ph * sizer.nu), (sizer.nu * sizer.ch)> Iz2uMat;
        mat<(sizer.nu * sizer.ch), (sizer.ph * sizer.nu)> Iu2zMat;
        mat<sizer.nu, sizer.nu> Sz2uMat;
        mat<sizer.nu, sizer.nu> Su2zMat;
    };
} // namespace mpc
