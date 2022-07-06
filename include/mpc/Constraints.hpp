#pragma once

#include <mpc/Base.hpp>

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
         * @brief Internal structure containing the value and the gradient
         * of the evaluated constraints.
         *
         * @tparam Tcon number of the constraints
         */
        template <int Tcon = Eigen::Dynamic>
        struct Cost
        {
            cvec<Tcon> value;
            cvec<(Tcon * ((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1))> grad;
        };

        Constraints() : Base<sizer>()
        {
            ctime = false;
        }

        ~Constraints() = default;

        /**
         * @brief Initialization hook override used to perform the
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed.
         */
        void onInit()
        {
            x0.resize(nx());
            Xmat.resize(ph() + 1, nx());
            Umat.resize(ph() + 1, nu());

            ceq.resize(ph() * nx());
            Jceq.resize((ph() * nx()) + (nu() * ch()) + 1, ph() * nx());

            cineq.resize(2 * (ph() * ny()));
            Jcineq.resize((ph() * nx()) + (nu() * ch()) + 1, 2 * (ph() * ny()));

            ceq_user.resize(eq());
            Jceq_user.resize((ph() * nx()) + (nu() * ch()) + 1, eq());

            cineq_user.resize(ineq());
            Jcineq_user.resize((ph() * nx()) + (nu() * ch()) + 1, ineq());
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
         * @brief Set if the provided dynamical model is in continuos time
         *
         * @param isContinuous system dynamics is defined in countinuos time
         * @param Ts discretization sample time, in general this is the inverse of the control loop frequency
         * @return true
         * @return false
         */
        bool setContinuos(bool isContinuous, double Ts = 0)
        {
            ts = Ts;
            ctime = isContinuous;
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
            const typename Base<sizer>::StateFunHandle handle)
        {
            checkOrQuit();
            return fUser = handle, true;
        }

        /**
         * @brief Set the system's output function (e.g. the state/output mapping)
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setOutputModel(
            const typename Base<sizer>::OutFunHandle handle)
        {
            checkOrQuit();
            return outUser = handle, true;
        }

        /**
         * @brief Set the user defined inequality constraints
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setIneqConstraints(
            const typename Base<sizer>::IConFunHandle handle)
        {
            checkOrQuit();
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
            const typename Base<sizer>::EConFunHandle handle)
        {
            checkOrQuit();
            return eqUser = handle, true;
        }

        /**
         * @brief Evaluate the user defined inequality constraints
         *
         * @param x internal optimization vector
         * @param hasGradient request the computation of the gradient
         * @return Cost<sizer.ineq> associated cost
         */
        Cost<sizer.ineq> evaluateIneq(
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            bool hasGradient)
        {
            checkOrQuit();

            mapping.unwrapVector(x, x0, Xmat, Umat, e);

            if (hasIneqConstraints())
            {
                // check if the output function of the system is defined
                // if so, let's compute the output along the horizon
                mat<(sizer.ph + 1), sizer.ny> Ymat;
                Ymat.resize(ph() + 1, ny());
                Ymat.setZero();

                if (hasOutputModel())
                {
                    for (size_t i = 0; i < ph() + 1; i++)
                    {
                        cvec<sizer.ny> YmatRow;
                        YmatRow.resize(ny());
                        YmatRow.setZero();

                        outUser(YmatRow, Xmat.row(i), Umat.row(i));
                        Ymat.row(i) = YmatRow;
                    }
                }

                ieqUser(cineq_user, Xmat, Ymat, Umat, e);

                mat<sizer.ineq, (sizer.ph * sizer.nx)> Jieqx;
                Jieqx.resize(ineq(), (ph() * nx()));

                mat<sizer.ineq, (sizer.ph * sizer.nu)> Jieqmv;
                Jieqmv.resize(ineq(), (ph() * nu()));

                cvec<sizer.ineq> Jie;
                Jie.resize(ineq());

                computeIneqJacobian(
                    Jieqx,
                    Jieqmv,
                    Jie,
                    Xmat,
                    Umat,
                    e,
                    cineq_user);

                glueJacobian<sizer.ineq>(
                    Jcineq_user,
                    Jieqx,
                    Jieqmv,
                    Jie);

                auto scaled_Jcineq_user = Jcineq_user;
                for (int j = 0; j < Jcineq_user.cols(); j++)
                {
                    int ioff = 0;
                    for (size_t k = 0; k < ph(); k++)
                    {
                        for (size_t ix = 0; ix < nx(); ix++)
                        {
                            scaled_Jcineq_user(ioff + ix, j) = scaled_Jcineq_user(ioff + ix, j) * mapping.StateScaling()(ix);
                        }
                        ioff = ioff + nx();
                    }
                }

                Jcineq_user = scaled_Jcineq_user;
            }
            else
            {
                cineq_user.setZero();
                Jcineq_user.setZero();
            }

            Cost<sizer.ineq> c;
            c.value = cineq_user;
            c.grad = Eigen::Map<cvec<(sizer.ineq * ((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1))>>(
                Jcineq_user.data(),
                Jcineq_user.size());

            Logger::instance().log(Logger::log_type::DETAIL)
                << "User inequality constraints value:\n"
                << std::setprecision(10)
                << c.value
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Gradient user inequality constraints not currently used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "User inequality constraints gradient:\n"
                    << std::setprecision(10)
                    << c.grad
                    << std::endl;
            }

            return c;
        }

        /**
         * @brief Evaluate the equality constraints for the system's dynamic
         *
         * @param x internal optimization vector
         * @param hasGradient request the computation of the gradient
         * @return Cost<(sizer.ph * sizer.nx)> associated cost
         */
        Cost<(sizer.ph * sizer.nx)> evaluateStateModelEq(
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            bool hasGradient)
        {
            checkOrQuit();
            mapping.unwrapVector(x, x0, Xmat, Umat, e);

            // Set MPC constraints
            getStateEqConstraints(hasGradient);

            Cost<(sizer.ph * sizer.nx)> c;
            c.value = ceq;
            c.grad = Eigen::Map<cvec<((sizer.ph * sizer.nx) * ((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1))>>(
                Jceq.data(),
                Jceq.size());

            Logger::instance().log(Logger::log_type::DETAIL)
                << "State equality constraints value:\n"
                << std::setprecision(10)
                << c.value
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "State equality constraints gradient not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "State equality constraints gradient:\n"
                    << std::setprecision(10)
                    << c.grad
                    << std::endl;
            }

            return c;
        }

        /**
         * @brief Evaluate the user defined equality constraints
         *
         * @param x internal optimization vector
         * @param hasGradient request the computation of the gradient
         * @return Cost<sizer.eq> associated cost
         */
        Cost<sizer.eq> evaluateEq(
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x,
            bool hasGradient)
        {
            checkOrQuit();
            mapping.unwrapVector(x, x0, Xmat, Umat, e);

            // Add user defined constraints
            if (hasEqConstraints())
            {
                eqUser(ceq_user, Xmat, Umat);

                mat<sizer.eq, (sizer.ph * sizer.nx)> Jeqx;
                Jeqx.resize(eq(), (ph() * nx()));

                mat<sizer.eq, (sizer.ph * sizer.nu)> Jeqmv;
                Jeqmv.resize(eq(), (ph() * nu()));

                computeEqJacobian(
                    Jeqx,
                    Jeqmv,
                    Xmat,
                    Umat,
                    ceq_user);

                glueJacobian<sizer.eq>(
                    Jceq_user,
                    Jeqx,
                    Jeqmv,
                    cvec<sizer.eq>::Zero(eq()));

                auto scaled_Jceq_user = Jceq_user;
                for (int j = 0; j < Jceq_user.cols(); j++)
                {
                    int ioff = 0;
                    for (size_t k = 0; k < ph(); k++)
                    {
                        for (size_t ix = 0; ix < nx(); ix++)
                        {
                            scaled_Jceq_user(ioff + ix, j) = scaled_Jceq_user(ioff + ix, j) * mapping.StateScaling()(ix);
                        }
                        ioff = ioff + nx();
                    }
                }

                Jceq_user = scaled_Jceq_user;
            }
            else
            {
                ceq_user.setZero();
                Jceq_user.setZero();
            }

            Cost<sizer.eq> c;
            c.value = ceq_user;
            c.grad = Eigen::Map<cvec<(sizer.eq * ((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1))>>(
                Jceq_user.data(),
                Jceq_user.size());

            Logger::instance().log(Logger::log_type::DETAIL)
                << "User equality constraints value:\n"
                << std::setprecision(10)
                << c.value
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Gradient user equality constraints not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "User equality constraints gradient:\n"
                    << std::setprecision(10)
                    << c.grad
                    << std::endl;
            }

            return c;
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
            mat<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1), Tnc> &Jres,
            const mat<Tnc, (sizer.ph * sizer.nx)> &Jstate,
            const mat<Tnc, (sizer.ph * sizer.nu)> &Jmanvar,
            const cvec<Tnc> &Jcon)
        {
            //#pragma omp parallel for
            for (size_t i = 0; i < ph(); i++)
            {
                Jres.middleRows(i * nx(), nx()) = Jstate.middleCols(i * nx(), nx()).transpose();
            }

            mat<Tnc, sizer.ph * sizer.nu> Jmanvar_mat;
            Jmanvar_mat.resize(Jres.cols(), ph() * nu());

            //#pragma omp parallel for
            for (size_t i = 0; i < ph(); i++)
            {
                Jmanvar_mat.block(0, i * nu(), Jres.cols(), nu()) = Jmanvar.middleCols(i * nu(), nu());
            }

            Jres.middleRows(ph() * nx(), nu() * ch()) = (Jmanvar_mat * mapping.Iz2u()).transpose();
            Jres.bottomRows(1) = Jcon.transpose();
        }

        /**
         * @brief Compute the internal state equality constraints penalty
         * and if requested the associated Jacobian matrix
         *
         * @param hasGradient request the computation of the gradient
         */
        void getStateEqConstraints(
            bool hasGradient)
        {
            ceq.setZero();
            Jceq.setZero();

            mat<(sizer.ph * sizer.nx), (sizer.ph * sizer.nx)> Jx;
            Jx.resize((ph() * nx()), (ph() * nx()));
            Jx.setZero();

            // TODO support measured noise
            mat<(sizer.ph * sizer.nx), (sizer.ph * sizer.nu)> Jmv;
            Jmv.resize((ph() * nx()), (ph() * nu()));
            Jmv.setZero();

            cvec<(sizer.ph * sizer.nx)> Je;
            Je.resize((ph() * nx()));
            Je.setZero();

            int ic = 0;

            mat<sizer.nx, sizer.nx> Ix;
            Ix.resize(nx(), nx());
            Ix.setIdentity(nx(), nx());

            mat<sizer.nx, sizer.nx> Sx, Tx;
            Sx = mapping.StateInverseScaling().asDiagonal();
            Tx = mapping.StateScaling().asDiagonal();

            // TODO bind for continuos time
            if (ctime)
            {
                //#pragma omp parallel for
                for (size_t i = 0; i < ph(); i++)
                {
                    cvec<sizer.nu> uk;
                    uk = Umat.row(i).transpose();
                    cvec<sizer.nx> xk;
                    xk = Xmat.row(i).transpose();

                    double h = ts / 2.0;
                    cvec<sizer.nx> xk1;
                    xk1 = Xmat.row(i + 1).transpose();

                    cvec<sizer.nx> fk;
                    fk.resize(nx());

                    fUser(fk, xk, uk);

                    cvec<sizer.nx> fk1;
                    fk1.resize(nx());

                    fUser(fk1, xk1, uk);

                    ceq.middleRows(ic, nx()) = xk + (h * (fk + fk1)) - xk1;
                    ceq.middleRows(ic, nx()) = ceq.middleRows(ic, nx()).array() / mapping.StateScaling().array();

                    if (hasGradient)
                    {
                        mat<sizer.nx, sizer.nx> Ak;
                        Ak.resize(nx(), nx());

                        mat<sizer.nx, sizer.nu> Bk;
                        Bk.resize(nx(), nu());

                        computeStateEqJacobian(Ak, Bk, fk, xk, uk);

                        mat<sizer.nx, sizer.nx> Ak1;
                        Ak1.resize(nx(), nx());

                        mat<sizer.nx, sizer.nu> Bk1;
                        Bk1.resize(nx(), nu());

                        computeStateEqJacobian(Ak1, Bk1, fk1, xk1, uk);

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
                //#pragma omp parallel for
                for (size_t i = 0; i < ph(); i++)
                {
                    cvec<sizer.nu> uk;
                    uk = Umat.row(i).transpose();
                    cvec<sizer.nx> xk;
                    xk = Xmat.row(i).transpose();

                    cvec<sizer.nx> xk1;
                    xk1.resize(nx());

                    fUser(xk1, xk, uk);

                    ceq.middleRows(ic, nx()) = Xmat.row(i + 1).transpose() - xk1;
                    ceq.middleRows(ic, nx()) = ceq.middleRows(ic, nx()).array() / mapping.StateScaling().array();

                    if (hasGradient)
                    {
                        mat<sizer.nx, sizer.nx> Ak;
                        Ak.resize(nx(), nx());

                        mat<sizer.nx, sizer.nu> Bk;
                        Bk.resize(nx(), nu());

                        computeStateEqJacobian(Ak, Bk, xk1, xk, uk);

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

            if (hasGradient)
            {
                glueJacobian<sizer.ph * sizer.nx>(Jceq, Jx, Jmv, Je);
            }
        }

        /**
         * @brief Approximate the user defined inequality constraints Jacobian matrices
         *
         * @param Jconx Jacobian matrix of the states-constraints
         * @param Jconmv Jacobian matrix of the input-constraints
         * @param Jcone Jacobian matrix of the slack-constraints
         * @param x0 current state configuration
         * @param u0 current optimal input configuration
         * @param e0 current slack value
         * @param f0 current user inequality constraints values
         */
        void computeIneqJacobian(
            mat<sizer.ineq, (sizer.ph * sizer.nx)> &Jconx,
            mat<sizer.ineq, (sizer.ph * sizer.nu)> &Jconmv,
            cvec<sizer.ineq> &Jcone,
            mat<(sizer.ph + 1), sizer.nx> x0,
            mat<(sizer.ph + 1), sizer.nu> u0,
            double e0, cvec<sizer.ineq> f0)
        {
            double dv = 1e-6;

            Jconx.setZero();

            // TODO support measured disturbaces
            Jconmv.setZero();

            Jcone.setZero();

            mat<(sizer.ph + 1), sizer.nx> Xa;
            Xa = x0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Xa.rows(); i++)
            {
                for (int j = 0; j < (int)Xa.cols(); j++)
                {
                    Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
                }
            }

            //#pragma omp parallel for
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    int ix = i + 1;
                    double dx = dv * Xa.array()(j);
                    x0(ix, j) = x0(ix, j) + dx;
                    cvec<sizer.ineq> f;
                    f.resize(ineq());

                    // check if the output function of the system is defined
                    // if so, let's compute the output along the horizon
                    mat<(sizer.ph + 1), sizer.ny> y0;
                    y0.resize(ph() + 1, ny());
                    y0.setZero();

                    if (hasOutputModel())
                    {
                        for (size_t k = 0; k < ph() + 1; k++)
                        {
                            mpc::cvec<sizer.ny> y0Row;
                            y0Row.resize(ny());
                            y0Row.setZero();

                            outUser(y0Row, x0.row(k), u0.row(k));
                            y0.row(k) = y0Row;
                        }
                    }

                    ieqUser(f, x0, y0, u0, e);
                    x0(ix, j) = x0(ix, j) - dx;
                    cvec<sizer.ineq> df;
                    df = (f - f0) / dx;
                    Jconx.middleCols(i * nx(), nx()).col(j) = df;
                }
            }

            mat<(sizer.ph + 1), sizer.nu> Ua;
            Ua = u0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Ua.rows(); i++)
            {
                for (int j = 0; j < (int)Ua.cols(); j++)
                {
                    Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
                }
            }

            //#pragma omp parallel for
            for (size_t i = 0; i < (ph() - 1); i++)
                // TODO support measured disturbaces
                for (size_t j = 0; j < nu(); j++)
                {
                    int k = j;
                    double du = dv * Ua.array()(k);
                    u0(i, k) = u0(i, k) + du;
                    cvec<sizer.ineq> f;
                    f.resize(ineq());

                    // check if the output function of the system is defined
                    // if so, let's compute the output along the horizon
                    mat<(sizer.ph + 1), sizer.ny> y0;
                    y0.resize(ph() + 1, ny());
                    y0.setZero();

                    if (hasOutputModel())
                    {
                        for (size_t k = 0; k < ph() + 1; k++)
                        {
                            mpc::cvec<sizer.ny> y0Row;
                            y0Row.resize(ny());
                            y0Row.setZero();

                            outUser(y0Row, x0.row(k), u0.row(k));
                            y0.row(k) = y0Row;
                        }
                    }

                    ieqUser(f, x0, y0, u0, e);
                    u0(i, k) = u0(i, k) - du;
                    cvec<sizer.ineq> df;
                    df = (f - f0) / du;
                    Jconmv.middleCols(i * nu(), nu()).col(j) = df;
                }

            // TODO support measured disturbaces
            //#pragma omp parallel for
            for (size_t j = 0; j < nu(); j++)
            {
                int k = j;
                double du = dv * Ua.array()(k);
                u0((ph() - 1), k) = u0((ph() - 1), k) + du;
                u0(ph(), k) = u0(ph(), k) + du;
                cvec<sizer.ineq> f;
                f.resize(ineq());

                // check if the output function of the system is defined
                // if so, let's compute the output along the horizon
                mat<(sizer.ph + 1), sizer.ny> y0;
                y0.resize(ph() + 1, ny());
                y0.setZero();

                if (hasOutputModel())
                {
                    for (size_t k = 0; k < ph() + 1; k++)
                    {
                        mpc::cvec<sizer.ny> y0Row;
                        y0Row.resize(ny());
                        y0Row.setZero();

                        outUser(y0Row, x0.row(k), u0.row(k));
                        y0.row(k) = y0Row;
                    }
                }

                ieqUser(f, x0, y0, u0, e);
                u0((ph() - 1), k) = u0((ph() - 1), k) - du;
                u0(ph(), k) = u0(ph(), k) - du;
                cvec<sizer.ineq> df;
                df = (f - f0) / du;
                Jconmv.middleCols(((ph() - 1) * nu()), nu()).col(j) = df;
            }

            double ea = fmax(1e-6, abs(e0));
            double de = ea * dv;
            cvec<sizer.ineq> f1;
            f1.resize(ineq());

            // check if the output function of the system is defined
            // if so, let's compute the output along the horizon
            mat<(sizer.ph + 1), sizer.ny> y0;
            y0.resize(ph() + 1, ny());
            y0.setZero();

            if (hasOutputModel())
            {
                for (size_t i = 0; i < ph() + 1; i++)
                {
                    mpc::cvec<sizer.ny> y0Row;
                    y0Row.resize(ny());
                    y0Row.setZero();

                    outUser(y0Row, x0.row(i), u0.row(i));
                    y0.row(i) = y0Row;
                }
            }

            ieqUser(f1, x0, y0, u0, e0 + de);

            cvec<sizer.ineq> f2;
            f2.resize(ineq());

            // check if the output function of the system is defined
            // if so, let's compute the output along the horizon
            y0.setZero();

            if (hasOutputModel())
            {
                for (size_t i = 0; i < ph() + 1; i++)
                {
                    mpc::cvec<sizer.ny> y0Row;
                    y0Row.resize(ny());
                    y0Row.setZero();

                    outUser(y0Row, x0.row(i), u0.row(i));
                    y0.row(i) = y0Row;
                }
            }

            ieqUser(f2, x0, y0, u0, e0 - de);
            Jcone = (f1 - f2) / (2 * de);
        }

        /**
         * @brief Approximate the user defined equality constraints Jacobian matrices
         *
         * @param Jconx Jacobian matrix of the states-constraints
         * @param Jconmv Jacobian matrix of the input-constraints
         * @param x0 current state configuration
         * @param u0 current optimal input configuration
         * @param f0 current user equality constraints values
         */
        void computeEqJacobian(
            mat<sizer.eq, (sizer.ph * sizer.nx)> &Jconx,
            mat<sizer.eq, (sizer.ph * sizer.nu)> &Jconmv,
            mat<(sizer.ph + 1), sizer.nx> x0,
            mat<(sizer.ph + 1), sizer.nu> u0, cvec<sizer.eq> f0)
        {
            double dv = 1e-6;

            Jconx.setZero();

            // TODO support measured disturbaces
            Jconmv.setZero();

            mat<(sizer.ph + 1), sizer.nx> Xa;
            Xa = x0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Xa.rows(); i++)
            {
                for (int j = 0; j < (int)Xa.cols(); j++)
                {
                    Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
                }
            }

            //#pragma omp parallel for
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    int ix = i + 1;
                    double dx = dv * Xa.array()(j);
                    x0(ix, j) = x0(ix, j) + dx;
                    cvec<sizer.eq> f;
                    f.resize(eq());
                    eqUser(f, x0, u0);
                    x0(ix, j) = x0(ix, j) - dx;
                    cvec<sizer.eq> df;
                    df = (f - f0) / dx;
                    Jconx.middleCols(i * nx(), nx()).col(j) = df;
                }
            }

            mat<(sizer.ph + 1), sizer.nu> Ua;
            Ua = u0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Ua.rows(); i++)
            {
                for (int j = 0; j < (int)Ua.cols(); j++)
                {
                    Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
                }
            }

            //#pragma omp parallel for
            for (size_t i = 0; i < (ph() - 1); i++)
            {
                // TODO support measured disturbaces
                for (size_t j = 0; j < nu(); j++)
                {
                    int k = j;
                    double du = dv * Ua.array()(k);
                    u0(i, k) = u0(i, k) + du;
                    cvec<sizer.eq> f;
                    f.resize(eq());
                    eqUser(f, x0, u0);
                    u0(i, k) = u0(i, k) - du;
                    cvec<sizer.eq> df;
                    df = (f - f0) / du;
                    Jconmv.middleCols(i * nu(), nu()).col(j) = df;
                }
            }

            // TODO support measured disturbaces
            //#pragma omp parallel for
            for (size_t j = 0; j < nu(); j++)
            {
                int k = j;
                double du = dv * Ua.array()(k);
                u0((ph() - 1), k) = u0((ph() - 1), k) + du;
                u0(ph(), k) = u0(ph(), k) + du;
                cvec<sizer.eq> f;
                f.resize(eq());
                eqUser(f, x0, u0);
                u0((ph() - 1), k) = u0((ph() - 1), k) - du;
                u0(ph(), k) = u0(ph(), k) - du;
                cvec<sizer.eq> df;
                df = (f - f0) / du;
                Jconmv.middleCols(((ph() - 1) * nu()), nu()).col(j) = df;
            }
        }

        /**
         * @brief Approximate the system's dynamics equality constraints Jacobian matrices
         *
         * @param Jconx Jacobian matrix of the states-constraints
         * @param Jmv Jacobian matrix of the states-inputs
         * @param f0 current user equality constraints values
         * @param x0 current state configuration
         * @param u0 current optimal input configuration
         */
        void computeStateEqJacobian(
            mat<sizer.nx, sizer.nx> &Jx,
            mat<sizer.nx, sizer.nu> &Jmv,
            cvec<sizer.nx> f0,
            cvec<sizer.nx> x0,
            cvec<sizer.nu> u0)
        {
            Jx.setZero();
            Jmv.setZero();

            double dv = 1e-6;

            cvec<sizer.nx> Xa;
            Xa = x0.cwiseAbs();
            //#pragma omp parallel for
            for (size_t i = 0; i < nx(); i++)
            {
                Xa(i) = (Xa(i) < 1) ? 1 : Xa(i);
            }

            //#pragma omp parallel for
            for (size_t i = 0; i < nx(); i++)
            {
                double dx = dv * Xa(i);
                x0(i) = x0(i) + dx;
                cvec<sizer.nx> f;
                f.resize(nx());
                fUser(f, x0, u0);
                x0(i) = x0(i) - dx;
                cvec<sizer.nx> df;
                df = (f - f0) / dx;
                Jx.block(0, i, nx(), 1) = df;
            }

            cvec<sizer.nu> Ua = u0.cwiseAbs();
            //#pragma omp parallel for
            for (size_t i = 0; i < nu(); i++)
            {
                Ua(i) = (Ua(i) < 1) ? 1 : Ua(i);
            }

            //#pragma omp parallel for
            for (size_t i = 0; i < nu(); i++)
            {
                // TODO support measured disturbaces
                int k = i;
                double du = dv * Ua(k);
                u0(k) = u0(k) + du;
                cvec<sizer.nx> f;
                f.resize(nx());
                fUser(f, x0, u0);
                u0(k) = u0(k) - du;
                cvec<sizer.nx> df;
                df = (f - f0) / du;
                Jmv.block(0, i, nx(), 1) = df;
            }
        }

        bool ctime;

        cvec<sizer.ph * sizer.nx> ceq;
        mat<(sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1, sizer.ph * sizer.nx> Jceq;

        cvec<2 * sizer.ph * sizer.ny> cineq;
        mat<(sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1, 2 * sizer.ph * sizer.ny> Jcineq;

        cvec<sizer.eq> ceq_user;
        mat<(sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1, sizer.eq> Jceq_user;

        cvec<sizer.ineq> cineq_user;
        mat<(sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1, sizer.ineq> Jcineq_user;

        typename Base<sizer>::StateFunHandle fUser = nullptr;
        typename Base<sizer>::IConFunHandle ieqUser = nullptr;
        typename Base<sizer>::EConFunHandle eqUser = nullptr;
        typename Base<sizer>::OutFunHandle outUser = nullptr;

        using Base<sizer>::mapping;
        using Base<sizer>::x0;
        using Base<sizer>::Xmat;
        using Base<sizer>::Umat;
        using Base<sizer>::e;
        using Base<sizer>::ts;
        using Base<sizer>::niteration;
    };
} // namespace mpc
