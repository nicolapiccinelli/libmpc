/*
 *   Copyright (c) 2023-2025 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/NLMPC/Base.hpp>

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
    class Constraints : public Base<sizer>
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
         * @brief Internal structure containing the value and the Jacobian
         * of the evaluated constraints.
         *
         * @tparam Tcon number of the constraints
         */
        template <int Tcon = Eigen::Dynamic>
        struct Cost
        {
            cvec<Tcon> value;
            Eigen::Matrix<double, Tcon, ((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1), Eigen::RowMajor> jacobian;
            // mat<Tcon, ((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> jacobian;
        };

        Constraints() : Base<sizer>()
        {
        }

        ~Constraints() = default;

        /**
         * @brief Initialization hook override used to perform the
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed.
         */
        void onInit() override
        {
            COND_RESIZE_CVEC(sizer, x0, nx());
            COND_RESIZE_MAT(sizer, Xmat, ph() + 1, nx());
            COND_RESIZE_MAT(sizer, Umat, ph() + 1, nu());

            COND_RESIZE_CVEC(sizer, c_eq_sys.value, ph() * nx());
            COND_RESIZE_MAT(sizer, c_eq_sys.jacobian, ph() * nx(), ((ph() * nx()) + (nu() * ch()) + 1));

            COND_RESIZE_CVEC(sizer, c_eq_user_def.value, eq());
            COND_RESIZE_MAT(sizer, c_eq_user_def.jacobian, eq(), ((ph() * nx()) + (nu() * ch()) + 1));

            COND_RESIZE_CVEC(sizer, c_ineq_user_def.value, ineq());
            COND_RESIZE_MAT(sizer, c_ineq_user_def.jacobian, ineq(), ((ph() * nx()) + (nu() * ch()) + 1));
        }

        /**
         * @brief Return if the dynamical system has user defined inequality constraints
         *
         * @return true
         * @return false
         */
        bool hasIneqConstraints()
        {
            checkOrQuit();
            return ieqUser != nullptr;
        }

        /**
         * @brief Return if the dynamical system has user defined equality constraints
         *
         * @return true
         * @return false
         */
        bool hasEqConstraints()
        {
            checkOrQuit();
            return eqUser != nullptr;
        }

        /**
         * @brief Set the user defined inequality constraints
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setIneqConstraints(
            const typename Base<sizer>::IConFunHandle handle, const float& tolerance)
        {
            checkOrQuit();

            // if the number of inequality constraints is zero let's avoid
            // the definition of the user inequality constraints
            if (ineq() == 0)
            {
                return false;
            }

            ieq_tolerance = tolerance;
            return ieqUser = handle, true;
        }

        /**
         * @brief Set the user defined equality constraints
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setEqConstraints(
            const typename Base<sizer>::EConFunHandle handle, const float& tolerance)
        {
            checkOrQuit();

            // if the number of equality constraints is zero let's avoid
            // the definition of the user equality constraints
            if (eq() == 0)
            {
                return false;
            }

            eq_tolerance = tolerance;
            return eqUser = handle, true;
        }

        /**
         * @brief Evaluate the feasibility of the optimization vector
         *
         * @param x optimization vector
         * @return true if the vector is feasible
         * @return false if the vector is not feasible
         */
        bool isFeasible(
            const Eigen::Matrix<double, (sizer.ph * sizer.nx + sizer.nu * sizer.ch + 1), 1> &x)
        {
            checkOrQuit();
            mapping->unwrapVector(x, x0, Xmat, Umat, e);

            if (hasIneqConstraints())
            {
                Eigen::Matrix<double, sizer.ph + 1, sizer.ny> Ymat = model->getOutput(Xmat, Umat);
                ieqUser(c_ineq_user_def.value, Xmat, Ymat, Umat, e);

                // print the vector of inequality constraints
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "User inequality constraints value (feasibility):\n"
                    << std::setprecision(10)
                    << c_ineq_user_def.value
                    << std::endl;

                // Check if all inequality constraints are satisfied (<= 0)
                if ((c_ineq_user_def.value.array() > ieq_tolerance).any())
                {
                    return false;
                }

            }

            if (hasEqConstraints())
            {
                eqUser(c_eq_user_def.value, Xmat, Umat);

                // print the vector of equality constraints
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "User equality constraints value (feasibility):\n"
                    << std::setprecision(10)
                    << c_eq_user_def.value
                    << std::endl;

                // Check if all equality constraints are satisfied (== 0)
                if (c_eq_user_def.value.array().abs().maxCoeff() > eq_tolerance)
                {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Evaluate the user defined inequality constraints
         *
         * @param x internal optimization vector
         * @param hasJacobian request the computation of the jacobian
         * @return Cost<sizer.ineq> associated cost
         */
        Cost<sizer.ineq> &evaluateIneq(
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            bool hasJacobian)
        {
            checkOrQuit();

            mapping->unwrapVector(x, x0, Xmat, Umat, e);

            if (hasIneqConstraints())
            {
                Logger::instance().log(Logger::LogType::DETAIL) << "User inequality constraints detected" << std::endl;

                // check if the output function of the system is defined
                // if so, let's compute the output along the horizon
                mat<(sizer.ph + 1), sizer.ny> Ymat = model->getOutput(Xmat, Umat);
                ieqUser(c_ineq_user_def.value, Xmat, Ymat, Umat, e);

                mat<sizer.ineq, (sizer.ph * sizer.nx)> Jieqx;
                COND_RESIZE_MAT(sizer, Jieqx, ineq(), (ph() * nx()));

                mat<sizer.ineq, (sizer.ph * sizer.nu)> Jieqmv;
                COND_RESIZE_MAT(sizer, Jieqmv, ineq(), (ph() * nu()));

                cvec<sizer.ineq> Jie;
                COND_RESIZE_CVEC(sizer, Jie, ineq());

                computeIneqJacobian(
                    Jieqx,
                    Jieqmv,
                    Jie,
                    Xmat,
                    Umat,
                    e);

                Logger::instance().log(Logger::LogType::DETAIL)
                    << "User inequality state constraints jacobian:\n"
                    << std::setprecision(10)
                    << Jieqx
                    << std::endl;

                Logger::instance().log(Logger::LogType::DETAIL)
                    << "User inequality inputs constraints jacobian:\n"
                    << std::setprecision(10)
                    << Jieqmv
                    << std::endl;

                Logger::instance().log(Logger::LogType::DETAIL)
                    << "User inequality slack constraints jacobian:\n"
                    << std::setprecision(10)
                    << Jie
                    << std::endl;

                glueJacobian<sizer.ineq>(
                    c_ineq_user_def.jacobian,
                    Jieqx,
                    Jieqmv,
                    Jie);

                auto scaled_Jcineq_user = c_ineq_user_def.jacobian;
                for (int j = 0; j < c_ineq_user_def.jacobian.rows(); j++)
                {
                    int ioff = 0;
                    for (size_t k = 0; k < ph(); k++)
                    {
                        for (size_t ix = 0; ix < nx(); ix++)
                        {
                            scaled_Jcineq_user(j,ioff + ix) = scaled_Jcineq_user(j,ioff + ix) * mapping->StateScaling()(ix);
                        }

                        ioff = ioff + nx();
                    }
                }

                c_ineq_user_def.jacobian = scaled_Jcineq_user;
            }
            else
            {
                Logger::instance().log(Logger::LogType::DETAIL) << "No user inequality constraints detected" << std::endl;

                c_ineq_user_def.value.setZero();
                c_ineq_user_def.jacobian.setZero();
            }

            Logger::instance().log(Logger::LogType::DETAIL)
                << "User inequality constraints value:\n"
                << std::setprecision(10)
                << c_ineq_user_def.value
                << std::endl;

            if (!hasJacobian)
            {
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "Jacobian user inequality constraints not currently used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "User inequality constraints jacobian:\n"
                    << std::setprecision(10)
                    << c_ineq_user_def.jacobian
                    << std::endl;
            }

            return c_ineq_user_def;
        }

        /**
         * @brief Evaluate the equality constraints for the system's dynamic
         *
         * @param x internal optimization vector
         * @param hasJacobian request the computation of the jacobian
         * @return Cost<(sizer.ph * sizer.nx)> associated cost
         */
        Cost<(sizer.ph * sizer.nx)> &evaluateStateModelEq(
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            bool hasJacobian)
        {
            checkOrQuit();
            mapping->unwrapVector(x, x0, Xmat, Umat, e);

            // Set MPC constraints
            getStateEqConstraints(hasJacobian);

            Logger::instance().log(Logger::LogType::DETAIL)
                << "State equality constraints value:\n"
                << std::setprecision(10)
                << c_eq_sys.value
                << std::endl;
            if (!hasJacobian)
            {
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "State equality constraints jacobian not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "State equality constraints jacobian:\n"
                    << std::setprecision(10)
                    << c_eq_sys.jacobian
                    << std::endl;
            }

            return c_eq_sys;
        }

        /**
         * @brief Evaluate the user defined equality constraints
         *
         * @param x internal optimization vector
         * @param hasJacobian request the computation of the jacobian
         * @return Cost<sizer.eq> associated cost
         */
        Cost<sizer.eq> &evaluateEq(
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            bool hasJacobian)
        {
            checkOrQuit();
            mapping->unwrapVector(x, x0, Xmat, Umat, e);

            // Add user defined constraints
            if (hasEqConstraints())
            {
                Logger::instance().log(Logger::LogType::DETAIL) << "User equality constraints detected" << std::endl;

                eqUser(c_eq_user_def.value, Xmat, Umat);

                mat<sizer.eq, (sizer.ph * sizer.nx)> Jeqx;
                COND_RESIZE_MAT(sizer, Jeqx, eq(), (ph() * nx()));

                mat<sizer.eq, (sizer.ph * sizer.nu)> Jeqmv;
                COND_RESIZE_MAT(sizer, Jeqmv, eq(), (ph() * nu()));

                computeEqJacobian(
                    Jeqx,
                    Jeqmv,
                    Xmat,
                    Umat);

                glueJacobian<sizer.eq>(
                    c_eq_user_def.jacobian,
                    Jeqx,
                    Jeqmv,
                    cvec<sizer.eq>::Zero(eq()));

                auto scaled_Jceq_user = c_eq_user_def.jacobian;
                for (int j = 0; j < c_eq_user_def.jacobian.rows(); j++)
                {
                    int ioff = 0;
                    for (size_t k = 0; k < ph(); k++)
                    {
                        for (size_t ix = 0; ix < nx(); ix++)
                        {
                            scaled_Jceq_user(j,ioff + ix) = scaled_Jceq_user(j,ioff + ix) * mapping->StateScaling()(ix);
                        }
                        ioff = ioff + nx();
                    }
                }

                c_eq_user_def.jacobian = scaled_Jceq_user;
            }
            else
            {
                Logger::instance().log(Logger::LogType::DETAIL) << "No user equality constraints detected" << std::endl;
                
                c_eq_user_def.value.setZero();
                c_eq_user_def.jacobian.setZero();
            }

            Logger::instance().log(Logger::LogType::DETAIL)
                << "User equality constraints value:\n"
                << std::setprecision(10)
                << c_eq_user_def.value
                << std::endl;
            if (!hasJacobian)
            {
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "Jacobian user equality constraints not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::LogType::DETAIL)
                    << "User equality constraints jacobian:\n"
                    << std::setprecision(10)
                    << c_eq_user_def.jacobian
                    << std::endl;
            }

            return c_eq_user_def;
        }

    private:
        /**
         * @brief Combines the Jacobian matrices of the system's dynamics,
         * the optimal control inputs and a set of constraints together
         *
         * @tparam Tnc number of constraints
         * @param Jres reference to the resulting Jacobian matrix
         * @param Jstate Jacobian matrix of the system's dynamics
         * @param Jmanvar Jacobian matrix of the optimal control inputs
         * @param Jcon Jacobian matrix of the constraint' set
         */
        template <int Tnc>
        void glueJacobian(
            Eigen::Matrix<double, Tnc, ((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1), Eigen::RowMajor> &Jres,
            const mat<Tnc, (sizer.ph * sizer.nx)> &Jstate,
            const mat<Tnc, (sizer.ph * sizer.nu)> &Jmanvar,
            const cvec<Tnc> &Jcon)
        {
            // #pragma omp parallel for
            for (size_t i = 0; i < ph(); i++)
            {
                Jres.middleCols(i * nx(), nx()) = Jstate.middleCols(i * nx(), nx());
            }

            mat<Tnc, sizer.ph * sizer.nu> Jmanvar_mat;
            COND_RESIZE_MAT(sizer, Jmanvar_mat, Jres.rows(), ph() * nu());

            // Fill the manipulated variable part
            // #pragma omp parallel for
            for (size_t i = 0; i < ph(); i++)
            {
                Jmanvar_mat.block(0, i * nu(), Jres.rows(), nu()) = Jmanvar.middleCols(i * nu(), nu());
            }

            Jres.middleCols(ph() * nx(), nu() * ch()) = (Jmanvar_mat * mapping->Iz2u());

            // Fill the constraints part (last column)
            Jres.middleCols((ph() * nx()) + (nu() * ch()), 1) = Jcon;
        }

        /**
         * @brief Compute the internal state equality constraints penalty
         * and if requested the associated Jacobian matrix
         *
         * @param hasJacobian request the computation of the jacobian
         */
        void getStateEqConstraints(
            bool hasJacobian)
        {
            c_eq_sys.value.setZero();
            c_eq_sys.jacobian.setZero();

            mat<(sizer.ph * sizer.nx), (sizer.ph * sizer.nx)> Jx;
            COND_RESIZE_MAT(sizer, Jx, (ph() * nx()), (ph() * nx()));
            Jx.setZero();

            // TODO support measured noise
            mat<(sizer.ph * sizer.nx), (sizer.ph * sizer.nu)> Jmv;
            COND_RESIZE_MAT(sizer, Jmv, (ph() * nx()), (ph() * nu()));
            Jmv.setZero();

            cvec<(sizer.ph * sizer.nx)> Je;
            COND_RESIZE_CVEC(sizer, Je, (ph() * nx()));
            Je.setZero();

            int ic = 0;

            mat<sizer.nx, sizer.nx> Ix;
            COND_RESIZE_MAT(sizer, Ix, nx(), nx());
            Ix.setIdentity(nx(), nx());

            mat<sizer.nx, sizer.nx> Sx, Tx;
            Sx = mapping->StateInverseScaling().asDiagonal();
            Tx = mapping->StateScaling().asDiagonal();

            // TODO bind for continuous time
            if (model->isContinuousTime)
            {
                Logger::instance().log(Logger::LogType::DETAIL) << "Continuous time model detected, using finite differences" << std::endl;

                // #pragma omp parallel for
                for (size_t i = 0; i < ph(); i++)
                {
                    cvec<sizer.nu> uk;
                    uk = Umat.row(i).transpose();

                    cvec<sizer.nx> xk;
                    xk = Xmat.row(i).transpose();

                    double h = model->sampleTime / 2.0;
                    cvec<sizer.nx> xk1;
                    xk1 = Xmat.row(i + 1).transpose();

                    cvec<sizer.nx> fk, fk1;
                    COND_RESIZE_CVEC(sizer, fk, nx());
                    COND_RESIZE_CVEC(sizer, fk1, nx());

                    model->vectorField(fk, xk, uk, i);
                    model->vectorField(fk1, xk1, uk, i);

                    c_eq_sys.value.middleRows(ic, nx()) = xk + (h * (fk + fk1)) - xk1;
                    c_eq_sys.value.middleRows(ic, nx()) = c_eq_sys.value.middleRows(ic, nx()).array() / mapping->StateScaling().array();

                    if (hasJacobian)
                    {
                        mat<sizer.nx, sizer.nx> Ak;
                        COND_RESIZE_MAT(sizer, Ak, nx(), nx());

                        mat<sizer.nx, sizer.nu> Bk;
                        COND_RESIZE_MAT(sizer, Bk, nx(), nu());

                        computeStateEqJacobian(Ak, Bk, xk, uk, i);

                        mat<sizer.nx, sizer.nx> Ak1;
                        COND_RESIZE_MAT(sizer, Ak1, nx(), nx());

                        mat<sizer.nx, sizer.nu> Bk1;
                        COND_RESIZE_MAT(sizer, Bk1, nx(), nu());

                        computeStateEqJacobian(Ak1, Bk1, xk1, uk, i);

                        if (i > 0)
                        {
                            Jx.middleCols((i - 1) * nx(), nx()).middleRows(ic, nx()) = Ix + (h * Sx * Ak * Tx);
                        }

                        Jx.middleCols(i * nx(), nx()).middleRows(ic, nx()) = -Ix + (h * Sx * Ak1 * Tx);
                        Jmv.middleCols(i * nu(), nu()).middleRows(ic, nx()) = h * Sx * (Bk + Bk1);
                    }

                    ic += nx();
                }
            }
            else
            {
                Logger::instance().log(Logger::LogType::DETAIL) << "Discrete time model detected" << std::endl;

                // #pragma omp parallel for
                for (size_t i = 0; i < ph(); i++)
                {
                    cvec<sizer.nu> uk;
                    uk = Umat.row(i).transpose();
                    cvec<sizer.nx> xk;
                    xk = Xmat.row(i).transpose();

                    cvec<sizer.nx> xk1;
                    COND_RESIZE_CVEC(sizer, xk1, nx());

                    model->vectorField(xk1, xk, uk, i);

                    c_eq_sys.value.middleRows(ic, nx()) = Xmat.row(i + 1).transpose() - xk1;
                    c_eq_sys.value.middleRows(ic, nx()) = c_eq_sys.value.middleRows(ic, nx()).array() / mapping->StateScaling().array();

                    if (hasJacobian)
                    {
                        mat<sizer.nx, sizer.nx> Ak;
                        COND_RESIZE_MAT(sizer, Ak, nx(), nx());

                        mat<sizer.nx, sizer.nu> Bk;
                        COND_RESIZE_MAT(sizer, Bk, nx(), nu());

                        computeStateEqJacobian(Ak, Bk, xk, uk, i);

                        Ak = Sx * Ak * Tx;
                        Bk = Sx * Bk;

                        Jx.middleCols(i * nx(), nx()).middleRows(ic, nx()) = Ix;
                        if (i > 0)
                        {
                            Jx.middleCols((i - 1) * nx(), nx()).middleRows(ic, nx()) = -Ak;
                        }
                        Jmv.middleCols(i * nu(), nu()).middleRows(ic, nx()) = -Bk;
                    }

                    ic += nx();
                }
            }

            if (hasJacobian)
            {
                glueJacobian<sizer.ph * sizer.nx>(
                    c_eq_sys.jacobian,
                    Jx, Jmv, Je);
            }
        }

        /**
         * Computes the inequality Jacobian matrix for the given inputs.
         * The Jacobian is computed using the central difference method.
         *
         * @param Jconx The output matrix for the inequality Jacobian with respect to x.
         * @param Jconmv The output matrix for the inequality Jacobian with respect to u.
         * @param Jcone The output vector for the inequality Jacobian with respect to e.
         * @param x0 The input matrix representing the initial state trajectory.
         * @param u0 The input matrix representing the control trajectory.
         * @param e0 The input value representing the error.
         */
        void computeIneqJacobian(
            mat<sizer.ineq, (sizer.ph * sizer.nx)> &Jconx,
            mat<sizer.ineq, (sizer.ph * sizer.nu)> &Jconmv,
            cvec<sizer.ineq> &Jcone,
            mat<(sizer.ph + 1), sizer.nx> x0,
            mat<(sizer.ph + 1), sizer.nu> u0,
            double e0)
        {
            Jconx.setZero();
            Jconmv.setZero();
            Jcone.setZero();

            mat<(sizer.ph + 1), sizer.nx> Xa;
            Xa = x0.cwiseAbs().cwiseMax(1.0);

            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    int ix = i + 1;
                    double dx = dv * Xa.array()(j);
                    x0(ix, j) += dx;
                    cvec<sizer.ineq> f_plus, f_minus;
                    COND_RESIZE_CVEC(sizer, f_plus, ineq());
                    COND_RESIZE_CVEC(sizer, f_minus, ineq());

                    mat<(sizer.ph + 1), sizer.ny> y0_plus = model->getOutput(x0, u0);
                    ieqUser(f_plus, x0, y0_plus, u0, e0);

                    x0(ix, j) -= 2 * dx;
                    mat<(sizer.ph + 1), sizer.ny> y0_minus = model->getOutput(x0, u0);
                    ieqUser(f_minus, x0, y0_minus, u0, e0);

                    x0(ix, j) += dx;

                    cvec<sizer.ineq> df = (f_plus - f_minus) / (2 * dx);
                    Jconx.middleCols(i * nx(), nx()).col(j) = df;
                }
            }

            mat<(sizer.ph + 1), sizer.nu> Ua;
            Ua = u0.cwiseAbs().cwiseMax(1.0);

            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nu(); j++)
                {
                    double du = dv * Ua.array()(j);
                    u0(i, j) += du;
                    cvec<sizer.ineq> f_plus, f_minus;
                    COND_RESIZE_CVEC(sizer, f_plus, ineq());
                    COND_RESIZE_CVEC(sizer, f_minus, ineq());

                    mat<(sizer.ph + 1), sizer.ny> y0_plus = model->getOutput(x0, u0);
                    ieqUser(f_plus, x0, y0_plus, u0, e0);

                    u0(i, j) -= 2 * du;
                    mat<(sizer.ph + 1), sizer.ny> y0_minus = model->getOutput(x0, u0);
                    ieqUser(f_minus, x0, y0_minus, u0, e0);

                    u0(i, j) += du;

                    cvec<sizer.ineq> df = (f_plus - f_minus) / (2 * du);
                    Jconmv.middleCols(i * nu(), nu()).col(j) = df;
                }
            }

            double ea = fmax(dv, fabs(e0));
            double de = ea * dv;
            cvec<sizer.ineq> f1, f2;
            COND_RESIZE_CVEC(sizer, f1, ineq());
            COND_RESIZE_CVEC(sizer, f2, ineq());

            mat<(sizer.ph + 1), sizer.ny> y0 = model->getOutput(x0, u0);
            ieqUser(f1, x0, y0, u0, e0 + de);

            y0 = model->getOutput(x0, u0);
            ieqUser(f2, x0, y0, u0, e0 - de);

            Jcone = (f1 - f2) / (2 * de);
        }

        /**
         * Computes the Jacobian matrix of equality constraints using the central difference method.
         *
         * @param Jconx The output matrix for the Jacobian of equality constraints with respect to the state variables.
         * @param Jconmv The output matrix for the Jacobian of equality constraints with respect to the control variables.
         * @param x0 The initial state vector.
         * @param u0 The initial control vector.
         */
        void computeEqJacobian(
            mat<sizer.eq, (sizer.ph * sizer.nx)> &Jconx,
            mat<sizer.eq, (sizer.ph * sizer.nu)> &Jconmv,
            mat<(sizer.ph + 1), sizer.nx> x0,
            mat<(sizer.ph + 1), sizer.nu> u0)
        {
            Jconx.setZero();
            Jconmv.setZero();

            mat<(sizer.ph + 1), sizer.nx> Xa;
            Xa = x0.cwiseAbs().cwiseMax(1.0);

            // Compute Jconx using central difference method
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    int ix = i + 1;
                    // Calculate perturbation
                    double dx = dv * Xa(ix, j);
                    // Forward perturbation
                    x0(ix, j) += dx;
                    cvec<sizer.eq> f_plus;
                    COND_RESIZE_CVEC(sizer, f_plus, eq());
                    // Compute equality constraints with perturbed state
                    eqUser(f_plus, x0, u0);
                    // Backward perturbation
                    x0(ix, j) -= 2 * dx;
                    cvec<sizer.eq> f_minus;
                    COND_RESIZE_CVEC(sizer, f_minus, eq());
                    // Compute equality constraints with perturbed state
                    eqUser(f_minus, x0, u0);
                    // Restore original state
                    x0(ix, j) += dx;
                    // Compute central difference
                    cvec<sizer.eq> df = (f_plus - f_minus) / (2 * dx);
                    Jconx.middleCols(i * nx(), nx()).col(j) = df;
                }
            }

            mat<(sizer.ph + 1), sizer.nu> Ua;
            Ua = u0.cwiseAbs().cwiseMax(1.0);

            // Compute Jconmv using central difference method
            for (size_t i = 0; i < (ph() - 1); i++)
            {
                for (size_t j = 0; j < nu(); j++)
                {
                    // Calculate perturbation
                    double du = dv * Ua(ph() - 1, j);
                    // Forward perturbation
                    u0(i, j) += du;
                    cvec<sizer.eq> f_plus;
                    COND_RESIZE_CVEC(sizer, f_plus, eq());
                    // Compute equality constraints with perturbed control
                    eqUser(f_plus, x0, u0);
                    // Backward perturbation
                    u0(i, j) -= 2 * du;
                    cvec<sizer.eq> f_minus;
                    COND_RESIZE_CVEC(sizer, f_minus, eq());
                    // Compute equality constraints with perturbed control
                    eqUser(f_minus, x0, u0);
                    // Restore original control
                    u0(i, j) += du;
                    // Compute central difference
                    cvec<sizer.eq> df = (f_plus - f_minus) / (2 * du);
                    // Update Jconmv
                    Jconmv.middleCols(i * nu(), nu()).col(j) = df;
                }
            }

            // Compute Jconmv for the last time step using central difference method
            for (size_t j = 0; j < nu(); j++)
            {
                // Calculate perturbation
                double du = dv * Ua(ph() - 1, j);
                // Forward perturbation
                u0((ph() - 1), j) += du;
                // Forward perturbation
                u0(ph(), j) += du;
                cvec<sizer.eq> f_plus;
                COND_RESIZE_CVEC(sizer, f_plus, eq());
                // Compute equality constraints with perturbed control
                eqUser(f_plus, x0, u0);
                // Backward perturbation
                u0((ph() - 1), j) -= 2 * du;
                // Backward perturbation
                u0(ph(), j) -= 2 * du;
                cvec<sizer.eq> f_minus;
                COND_RESIZE_CVEC(sizer, f_minus, eq());
                // Compute equality constraints with perturbed control
                eqUser(f_minus, x0, u0);
                // Restore original control
                u0((ph() - 1), j) += du;
                // Restore original control
                u0(ph(), j) += du;
                // Compute central difference
                cvec<sizer.eq> df = (f_plus - f_minus) / (2 * du);
                // Update Jconmv
                Jconmv.middleCols(((ph() - 1) * nu()), nu()).col(j) = df;
            }
        }

        /**
         * Computes the Jacobian matrices Jx and Jmv for the state equation.
         * The Jacobian matrices are computed using the central difference method.
         *
         * @param Jx   Reference to the matrix Jx where the computed Jacobian matrix for the state variables will be stored.
         * @param Jmv  Reference to the matrix Jmv where the computed Jacobian matrix for the control variables will be stored.
         * @param x0   Constant reference to the vector x0 representing the current state variables.
         * @param u0   Constant reference to the vector u0 representing the current control variables.
         * @param p    Unsigned integer representing the current parameter.
         */
        void computeStateEqJacobian(
            mat<sizer.nx, sizer.nx> &Jx,
            mat<sizer.nx, sizer.nu> &Jmv,
            cvec<sizer.nx> x0,
            cvec<sizer.nu> u0,
            unsigned int p)
        {
            Jx.setZero();
            Jmv.setZero();

            // this is computing the max(abs(x0), 1) for each
            // element of the state vector x0. This is then used
            // to scale the perturbation for each element of the state
            // vector.
            cvec<sizer.nx> Xa = x0.cwiseAbs().cwiseMax(1.0);

            // Compute Jx using central difference method
            for (size_t i = 0; i < nx(); i++)
            {
                double dx = dv * Xa(i);

                cvec<sizer.nx> x_plus = x0;
                cvec<sizer.nx> x_minus = x0;

                x_plus(i) += dx;
                x_minus(i) -= dx;

                cvec<sizer.nx> f_plus, f_minus;
                COND_RESIZE_CVEC(sizer, f_plus, nx());
                COND_RESIZE_CVEC(sizer, f_minus, nx());

                model->vectorField(f_plus, x_plus, u0, p);
                model->vectorField(f_minus, x_minus, u0, p);

                cvec<sizer.nx> df = (f_plus - f_minus) / (2 * dx);
                Jx.col(i) = df;
            }

            cvec<sizer.nu> Ua = u0.cwiseAbs().cwiseMax(1.0);

            // Compute Jmv using central difference method
            for (size_t i = 0; i < nu(); i++)
            {
                // TODO support measured disturbances
                double du = dv * Ua(i);
                cvec<sizer.nu> u_plus = u0;
                cvec<sizer.nu> u_minus = u0;

                u_plus(i) += du;
                u_minus(i) -= du;

                cvec<sizer.nx> f_plus, f_minus;
                COND_RESIZE_CVEC(sizer, f_plus, nx());
                COND_RESIZE_CVEC(sizer, f_minus, nx());

                model->vectorField(f_plus, x0, u_plus, p);
                model->vectorField(f_minus, x0, u_minus, p);

                cvec<sizer.nx> df = (f_plus - f_minus) / (2 * du);
                Jmv.col(i) = df;
            }
        }

        Cost<sizer.ph * sizer.nx> c_eq_sys;
        Cost<sizer.eq> c_eq_user_def;
        Cost<sizer.ineq> c_ineq_user_def;

        typename Base<sizer>::IConFunHandle ieqUser = nullptr;
        typename Base<sizer>::EConFunHandle eqUser = nullptr;

        using Base<sizer>::mapping;
        using Base<sizer>::model;
        using Base<sizer>::x0;
        using Base<sizer>::Xmat;
        using Base<sizer>::Umat;
        using Base<sizer>::e;
        using Base<sizer>::niteration;

        const double dv = sqrt(std::numeric_limits<double>::epsilon());
        double ieq_tolerance, eq_tolerance;
    };
} // namespace mpc