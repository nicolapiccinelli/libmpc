#pragma once

#include <mpc/Base.hpp>

namespace mpc {
template <
    int Tnx, int Tnu, int Tny,
    int Tph, int Tch,
    int Tineq, int Teq>
class Constraints : public Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq> {
private:
    using Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::checkOrQuit;
    using Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::dim;

public:
    template <int Tcon = Eigen::Dynamic>
    struct Cost {
        cvec<Tcon> value;
        cvec<(Dim<Tcon>() * ((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>()))> grad;
    };

    Constraints()
        : Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>()
    {
        ctime = false;
    }

    ~Constraints() = default;

    void onInit()
    {
        x0.resize(dim.nx.num());
        Xmat.resize(dim.ph.num() + 1, dim.nx.num());
        Umat.resize(dim.ph.num() + 1, dim.nu.num());

        ceq.resize(dim.ph.num() * dim.nx.num());
        Jceq.resize((dim.ph.num() * dim.nx.num()) + (dim.nu.num() * dim.ch.num()) + 1, dim.ph.num() * dim.nx.num());

        cineq.resize(2 * (dim.ph.num() * dim.ny.num()));
        Jcineq.resize((dim.ph.num() * dim.nx.num()) + (dim.nu.num() * dim.ch.num()) + 1, 2 * (dim.ph.num() * dim.ny.num()));

        ceq_user.resize(dim.eq.num());
        Jceq_user.resize((dim.ph.num() * dim.nx.num()) + (dim.nu.num() * dim.ch.num()) + 1, dim.eq.num());

        cineq_user.resize(dim.ineq.num());
        Jcineq_user.resize((dim.ph.num() * dim.nx.num()) + (dim.nu.num() * dim.ch.num()) + 1, dim.ineq.num());
    }

    bool hasOutputModel()
    {
        checkOrQuit();
        return outUser != nullptr;
    }

    bool hasIneqConstraints()
    {
        checkOrQuit();
        return ieqUser != nullptr;
    }

    bool hasEqConstraints()
    {
        checkOrQuit();
        return eqUser != nullptr;
    }

    bool setContinuos(bool isContinuous, double Ts = 0)
    {
        ts = Ts;
        ctime = isContinuous;
        return true;
    }

    bool setStateModel(
        const typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::StateFunHandle handle)
    {
        checkOrQuit();
        return fUser = handle, true;
    }

    bool setOutputModel(
        const typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::OutFunHandle handle)
    {
        checkOrQuit();
        return outUser = handle, true;
    }

    bool setIneqConstraints(
        const typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle handle)
    {
        checkOrQuit();
        return ieqUser = handle, true;
    }

    bool setEqConstraints(
        const typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle handle)
    {
        checkOrQuit();
        return eqUser = handle, true;
    }

    Cost<dim.ineq> evaluateIneq(
        cvec<((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>())> x,
        bool hasGradient)
    {
        checkOrQuit();

        mapping.unwrapVector(x, x0, Xmat, Umat, e);

        if (hasIneqConstraints()) {
            // check if the output function of the system is defined
            // if so, let's compute the output along the horizon
            mat<(dim.ph + Dim<1>()), Tny> Ymat;
            Ymat.setZero();
            if (hasOutputModel()) {
                outUser(Ymat, Xmat, Umat);
            }

            ieqUser(cineq_user, Xmat, Ymat, Umat, e);

            mat<dim.ineq, (dim.ph * dim.nx)> Jieqx;
            Jieqx.resize(dim.ineq.num(), (dim.ph.num() * dim.nx.num()));

            mat<dim.ineq, (dim.ph * dim.nu)> Jieqmv;
            Jieqmv.resize(dim.ineq.num(), (dim.ph.num() * dim.nu.num()));

            cvec<dim.ineq> Jie;
            Jie.resize(dim.ineq.num());

            computeIneqJacobian(
                Jieqx,
                Jieqmv,
                Jie,
                Xmat,
                Umat,
                e,
                cineq_user);

            glueJacobian<dim.ineq>(
                Jcineq_user,
                Jieqx,
                Jieqmv,
                Jie);

            auto scaled_Jcineq_user = Jcineq_user;
            for (int j = 0; j < Jcineq_user.cols(); j++) {
                int ioff = 0;
                for (size_t k = 0; k < dim.ph.num(); k++) {
                    for (size_t ix = 0; ix < dim.nx.num(); ix++) {
                        scaled_Jcineq_user(ioff + ix, j) = scaled_Jcineq_user(ioff + ix, j) * mapping.StateScaling()(ix);
                    }
                    ioff = ioff + dim.nx.num();
                }
            }

            Jcineq_user = scaled_Jcineq_user;
        } else {
            cineq_user.setZero();
            Jcineq_user.setZero();
        }

        Cost<Tineq> c;
        c.value = cineq_user;
        c.grad = Eigen::Map<cvec<(dim.ineq * ((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>()))>>(
            Jcineq_user.data(),
            Jcineq_user.size());

        Logger::instance().log(Logger::log_type::DETAIL)
            << "User inequality constraints value:\n"
            << std::setprecision(10)
            << c.value
            << std::endl;
        if (!hasGradient) {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "Gradient user inequality constraints not currently used"
                << std::endl;
        } else {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "User inequality constraints gradient:\n"
                << std::setprecision(10)
                << c.grad
                << std::endl;
        }

        return c;
    }

    Cost<(dim.ph * dim.nx)> evaluateStateModelEq(
        cvec<((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>())> x,
        bool hasGradient)
    {
        checkOrQuit();
        mapping.unwrapVector(x, x0, Xmat, Umat, e);

        // Set MPC constraints
        getStateEqConstraints(hasGradient);

        Cost<(dim.ph * dim.nx)> c;
        c.value = ceq;
        c.grad = Eigen::Map<cvec<((dim.ph * dim.nx) * ((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>()))>>(
            Jceq.data(),
            Jceq.size());

        Logger::instance().log(Logger::log_type::DETAIL)
            << "State equality constraints value:\n"
            << std::setprecision(10)
            << c.value
            << std::endl;
        if (!hasGradient) {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "State equality constraints gradient not currectly used"
                << std::endl;
        } else {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "State equality constraints gradient:\n"
                << std::setprecision(10)
                << c.grad
                << std::endl;
        }

        return c;
    }

    Cost<dim.eq> evaluateEq(
        cvec<((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>())> x,
        bool hasGradient)
    {
        checkOrQuit();
        mapping.unwrapVector(x, x0, Xmat, Umat, e);

        // Add user defined constraints
        if (hasEqConstraints()) {
            eqUser(ceq_user, Xmat, Umat);

            mat<dim.eq, (dim.ph * dim.nx)> Jeqx;
            Jeqx.resize(dim.eq.num(), (dim.ph.num() * dim.nx.num()));

            mat<dim.eq, (dim.ph * dim.nu)> Jeqmv;
            Jeqmv.resize(dim.eq.num(), (dim.ph.num() * dim.nu.num()));

            computeEqJacobian(
                Jeqx,
                Jeqmv,
                Xmat,
                Umat,
                ceq_user);

            glueJacobian<dim.eq>(
                Jceq_user,
                Jeqx,
                Jeqmv,
                cvec<dim.eq>::Zero(dim.eq.num()));

            auto scaled_Jceq_user = Jceq_user;
            for (int j = 0; j < Jceq_user.cols(); j++) {
                int ioff = 0;
                for (size_t k = 0; k < dim.ph.num(); k++) {
                    for (size_t ix = 0; ix < dim.nx.num(); ix++) {
                        scaled_Jceq_user(ioff + ix, j) = scaled_Jceq_user(ioff + ix, j) * mapping.StateScaling()(ix);
                    }
                    ioff = ioff + dim.nx.num();
                }
            }

            Jceq_user = scaled_Jceq_user;
        } else {
            ceq_user.setZero();
            Jceq_user.setZero();
        }

        Cost<dim.eq> c;
        c.value = ceq_user;
        c.grad = Eigen::Map<cvec<(dim.eq * ((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>()))>>(
            Jceq_user.data(),
            Jceq_user.size());

        Logger::instance().log(Logger::log_type::DETAIL)
            << "User equality constraints value:\n"
            << std::setprecision(10)
            << c.value
            << std::endl;
        if (!hasGradient) {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "Gradient user equality constraints not currectly used"
                << std::endl;
        } else {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "User equality constraints gradient:\n"
                << std::setprecision(10)
                << c.grad
                << std::endl;
        }

        return c;
    }

private:
    template <int Tnc>
    void glueJacobian(
        mat<((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>()).get(), Tnc>& Jres,
        const mat<Tnc, (dim.ph * dim.nx).get()>& Jstate,
        const mat<Tnc, (dim.ph * dim.nu).get()>& Jmanvar,
        const cvec<Tnc>& Jcon)
    {
        //#pragma omp parallel for
        for (size_t i = 0; i < dim.ph.num(); i++) {
            Jres.middleRows(i * dim.nx.num(), dim.nx.num()) = Jstate.middleCols(i * dim.nx.num(), dim.nx.num()).transpose();
        }

        mat<Tnc, dim.ph * dim.nu> Jmanvar_mat;
        Jmanvar_mat.resize(Jres.cols(), dim.ph.num() * dim.nu.num());

        //#pragma omp parallel for
        for (size_t i = 0; i < dim.ph.num(); i++) {
            Jmanvar_mat.block(0, i * dim.nu.num(), Jres.cols(), dim.nu.num()) = Jmanvar.middleCols(i * dim.nu.num(), dim.nu.num());
        }

        Jres.middleRows(dim.ph.num() * dim.nx.num(), dim.nu.num() * dim.ch.num()) = (Jmanvar_mat * mapping.Iz2u()).transpose();
        Jres.bottomRows(1) = Jcon.transpose();
    }

    void getStateEqConstraints(
        bool hasGradient)
    {
        ceq.setZero();
        Jceq.setZero();

        mat<(dim.ph * dim.nx), (dim.ph * dim.nx)> Jx;
        Jx.resize((dim.ph.num() * dim.nx.num()), (dim.ph.num() * dim.nx.num()));
        Jx.setZero();

        // TODO support measured noise
        mat<(dim.ph * dim.nx), (dim.ph * dim.nu)> Jmv;
        Jmv.resize((dim.ph.num() * dim.nx.num()), (dim.ph.num() * dim.nu.num()));
        Jmv.setZero();

        cvec<(dim.ph * dim.nx)> Je;
        Je.resize((dim.ph.num() * dim.nx.num()));
        Je.setZero();

        int ic = 0;

        mat<dim.nx, dim.nx> Ix;
        Ix.resize(dim.nx.num(), dim.nx.num());
        Ix.setIdentity(dim.nx.num(), dim.nx.num());

        mat<dim.nx, dim.nx> Sx, Tx;
        Sx = mapping.StateInverseScaling().asDiagonal();
        Tx = mapping.StateScaling().asDiagonal();

        // TODO bind for continuos time
        if (ctime) {
            //#pragma omp parallel for
            for (size_t i = 0; i < dim.ph.num(); i++) {
                cvec<dim.nu> uk;
                uk = Umat.row(i).transpose();
                cvec<dim.nx> xk;
                xk = Xmat.row(i).transpose();

                double h = ts / 2.0;
                cvec<dim.nx> xk1;
                xk1 = Xmat.row(i + 1).transpose();

                cvec<dim.nx> fk;
                fk.resize(dim.nx.num());

                fUser(fk, xk, uk);

                cvec<dim.nx> fk1;
                fk1.resize(dim.nx.num());

                fUser(fk1, xk1, uk);

                ceq.middleRows(ic, dim.nx.num()) = xk + (h * (fk + fk1)) - xk1;
                ceq.middleRows(ic, dim.nx.num()) = ceq.middleRows(ic, dim.nx.num()).array() / mapping.StateScaling().array();

                if (hasGradient) {
                    mat<dim.nx, dim.nx> Ak;
                    Ak.resize(dim.nx.num(), dim.nx.num());

                    mat<dim.nx, dim.nu> Bk;
                    Bk.resize(dim.nx.num(), dim.nu.num());

                    computeStateEqJacobian(Ak, Bk, fk, xk, uk);

                    mat<dim.nx, dim.nx> Ak1;
                    Ak1.resize(dim.nx.num(), dim.nx.num());

                    mat<dim.nx, dim.nu> Bk1;
                    Bk1.resize(dim.nx.num(), dim.nu.num());

                    computeStateEqJacobian(Ak1, Bk1, fk1, xk1, uk);

                    if (i > 0) {
                        Jx.middleCols((i - 1) * dim.nx.num(), dim.nx.num()).middleRows(ic, dim.nx.num()) = Ix + (h * Sx * Ak * Tx);
                    }

                    Jx.middleCols(i * dim.nx.num(), dim.nx.num()).middleRows(ic, dim.nx.num()) = -Ix + (h * Sx * Ak1 * Tx);
                    Jmv.middleCols(i * dim.nu.num(), dim.nu.num()).middleRows(ic, dim.nx.num()) = h * Sx * (Bk + Bk1);
                }

                ic += dim.nx.num();
            }
        } else {
            //#pragma omp parallel for
            for (size_t i = 0; i < dim.ph.num(); i++) {
                cvec<Tnu> uk;
                uk = Umat.row(i).transpose();
                cvec<Tnx> xk;
                xk = Xmat.row(i).transpose();

                cvec<Tnx> xk1;
                xk1.resize(dim.nx.num());

                fUser(xk1, xk, uk);

                ceq.middleRows(ic, dim.nx.num()) = Xmat.row(i + 1).transpose() - xk1;
                ceq.middleRows(ic, dim.nx.num()) = ceq.middleRows(ic, dim.nx.num()).array() / mapping.StateScaling().array();

                if (hasGradient) {
                    mat<dim.nx, dim.nx> Ak;
                    Ak.resize(dim.nx.num(), dim.nx.num());

                    mat<dim.nx, dim.nu> Bk;
                    Bk.resize(dim.nx.num(), dim.nu.num());

                    computeStateEqJacobian(Ak, Bk, xk1, xk, uk);

                    Ak = Sx * Ak * Tx;
                    Bk = Sx * Bk;

                    Jx.middleCols(i * dim.nx.num(), dim.nx.num()).middleRows(ic, dim.nx.num()) = Ix;
                    if (i > 0) {
                        Jx.middleCols((i - 1) * dim.nx.num(), dim.nx.num()).middleRows(ic, dim.nx.num()) = -Ak;
                    }
                    Jmv.middleCols(i * dim.nu.num(), dim.nu.num()).middleRows(ic, dim.nx.num()) = -Bk;
                }

                ic += dim.nx.num();
            }
        }

        if (hasGradient) {
            glueJacobian<dim.ph * dim.nx>(Jceq, Jx, Jmv, Je);
        }
    }

    void computeIneqJacobian(
        mat<Tineq, (dim.ph * dim.nx)>& Jconx,
        mat<Tineq, (dim.ph * dim.nu)>& Jconmv,
        cvec<Tineq>& Jcone,
        mat<(dim.ph + Dim<1>()), Tnx> x0,
        mat<(dim.ph + Dim<1>()), Tnu> u0,
        double e0, cvec<Tineq> f0)
    {
        double dv = 1e-6;

        Jconx.setZero();

        // TODO support measured disturbaces
        Jconmv.setZero();

        Jcone.setZero();

        mat<(dim.ph + Dim<1>()), dim.nx> Xa;
        Xa = x0.cwiseAbs();
        //#pragma omp parallel for
        for (int i = 0; i < (int)Xa.rows(); i++) {
            for (int j = 0; j < (int)Xa.cols(); j++) {
                Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < dim.ph.num(); i++) {
            for (size_t j = 0; j < dim.nx.num(); j++) {
                int ix = i + 1;
                double dx = dv * Xa.array()(j);
                x0(ix, j) = x0(ix, j) + dx;
                cvec<dim.ineq> f;
                f.resize(dim.ineq.num());

                // check if the output function of the system is defined
                // if so, let's compute the output along the horizon
                mat<(dim.ph + Dim<1>()), dim.ny> y0;
                y0.setZero();
                if (hasOutputModel()) {
                    outUser(y0, x0, u0);
                }

                ieqUser(f, x0, y0, u0, e);
                x0(ix, j) = x0(ix, j) - dx;
                cvec<dim.ineq> df;
                df = (f - f0) / dx;
                Jconx.middleCols(i * dim.nx.num(), dim.nx.num()).col(j) = df;
            }
        }

        mat<(dim.ph + Dim<1>()), dim.nu> Ua;
        Ua = u0.cwiseAbs();
        //#pragma omp parallel for
        for (int i = 0; i < (int)Ua.rows(); i++) {
            for (int j = 0; j < (int)Ua.cols(); j++) {
                Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < (dim.ph.num() - 1); i++)
            // TODO support measured disturbaces
            for (size_t j = 0; j < dim.nu.num(); j++) {
                int k = j;
                double du = dv * Ua.array()(k);
                u0(i, k) = u0(i, k) + du;
                cvec<dim.ineq> f;
                f.resize(dim.ineq.num());

                // check if the output function of the system is defined
                // if so, let's compute the output along the horizon
                mat<(dim.ph + Dim<1>()), dim.ny> y0;
                y0.setZero();
                if (hasOutputModel()) {
                    outUser(y0, x0, u0);
                }

                ieqUser(f, x0, y0, u0, e);
                u0(i, k) = u0(i, k) - du;
                cvec<dim.ineq> df;
                df = (f - f0) / du;
                Jconmv.middleCols(i * dim.nu.num(), dim.nu.num()).col(j) = df;
            }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t j = 0; j < dim.nu.num(); j++) {
            int k = j;
            double du = dv * Ua.array()(k);
            u0((dim.ph.num() - 1), k) = u0((dim.ph.num() - 1), k) + du;
            u0(dim.ph.num(), k) = u0(dim.ph.num(), k) + du;
            cvec<dim.ineq> f;
            f.resize(dim.ineq.num());

            // check if the output function of the system is defined
            // if so, let's compute the output along the horizon
            mat<(dim.ph + Dim<1>()), Tny> y0;
            y0.setZero();
            if (hasOutputModel()) {
                outUser(y0, x0, u0);
            }

            ieqUser(f, x0, y0, u0, e);
            u0((dim.ph.num() - 1), k) = u0((dim.ph.num() - 1), k) - du;
            u0(dim.ph.num(), k) = u0(dim.ph.num(), k) - du;
            cvec<dim.ineq> df;
            df = (f - f0) / du;
            Jconmv.middleCols(((dim.ph.num() - 1) * dim.nu.num()), dim.nu.num()).col(j) = df;
        }

        double ea = fmax(1e-6, abs(e0));
        double de = ea * dv;
        cvec<dim.ineq> f1;
        f1.resize(dim.ineq.num());

        // check if the output function of the system is defined
        // if so, let's compute the output along the horizon
        mat<(dim.ph + Dim<1>()), dim.ny> y0;
        y0.setZero();
        if (hasOutputModel()) {
            outUser(y0, x0, u0);
        }

        ieqUser(f1, x0, y0, u0, e0 + de);

        cvec<dim.ineq> f2;
        f2.resize(dim.ineq.num());

        // check if the output function of the system is defined
        // if so, let's compute the output along the horizon
        y0.setZero();
        if (hasOutputModel()) {
            outUser(y0, x0, u0);
        }

        ieqUser(f2, x0, y0, u0, e0 - de);
        Jcone = (f1 - f2) / (2 * de);
    }

    void computeEqJacobian(
        mat<dim.eq, (dim.ph * dim.nx)>& Jconx,
        mat<dim.eq, (dim.ph * dim.nu)>& Jconmv,
        mat<(dim.ph + Dim<1>()), dim.nx> x0,
        mat<(dim.ph + Dim<1>()), Tnu> u0, cvec<dim.eq> f0)
    {
        double dv = 1e-6;

        Jconx.setZero();

        // TODO support measured disturbaces
        Jconmv.setZero();

        mat<(dim.ph + Dim<1>()), dim.nx> Xa;
        Xa = x0.cwiseAbs();
        //#pragma omp parallel for
        for (int i = 0; i < (int)Xa.rows(); i++) {
            for (int j = 0; j < (int)Xa.cols(); j++) {
                Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < dim.ph.num(); i++) {
            for (size_t j = 0; j < dim.nx.num(); j++) {
                int ix = i + 1;
                double dx = dv * Xa.array()(j);
                x0(ix, j) = x0(ix, j) + dx;
                cvec<dim.eq> f;
                f.resize(dim.eq.num());
                eqUser(f, x0, u0);
                x0(ix, j) = x0(ix, j) - dx;
                cvec<dim.eq> df;
                df = (f - f0) / dx;
                Jconx.middleCols(i * dim.nx.num(), dim.nx.num()).col(j) = df;
            }
        }

        mat<(dim.ph + Dim<1>()), Tnu> Ua;
        Ua = u0.cwiseAbs();
        //#pragma omp parallel for
        for (int i = 0; i < (int)Ua.rows(); i++) {
            for (int j = 0; j < (int)Ua.cols(); j++) {
                Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < (dim.ph.num() - 1); i++) {
            // TODO support measured disturbaces
            for (size_t j = 0; j < dim.nu.num(); j++) {
                int k = j;
                double du = dv * Ua.array()(k);
                u0(i, k) = u0(i, k) + du;
                cvec<dim.eq> f;
                f.resize(dim.eq.num());
                eqUser(f, x0, u0);
                u0(i, k) = u0(i, k) - du;
                cvec<dim.eq> df;
                df = (f - f0) / du;
                Jconmv.middleCols(i * dim.nu.num(), dim.nu.num()).col(j) = df;
            }
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t j = 0; j < dim.nu.num(); j++) {
            int k = j;
            double du = dv * Ua.array()(k);
            u0((dim.ph.num() - 1), k) = u0((dim.ph.num() - 1), k) + du;
            u0(dim.ph.num(), k) = u0(dim.ph.num(), k) + du;
            cvec<dim.eq> f;
            f.resize(dim.eq.num());
            eqUser(f, x0, u0);
            u0((dim.ph.num() - 1), k) = u0((dim.ph.num() - 1), k) - du;
            u0(dim.ph.num(), k) = u0(dim.ph.num(), k) - du;
            cvec<dim.eq> df;
            df = (f - f0) / du;
            Jconmv.middleCols(((dim.ph.num() - 1) * dim.nu.num()), dim.nu.num()).col(j) = df;
        }
    }

    void computeStateEqJacobian(
        mat<Tnx, Tnx>& Jx,
        mat<Tnx, Tnu>& Jmv,
        cvec<Tnx> f0,
        cvec<Tnx> x0,
        cvec<Tnu> u0)
    {
        Jx.setZero();
        Jmv.setZero();

        double dv = 1e-6;

        cvec<dim.nx> Xa;
        Xa = x0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < dim.nx.num(); i++) {
            Xa(i) = (Xa(i) < 1) ? 1 : Xa(i);
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < dim.nx.num(); i++) {
            double dx = dv * Xa(i);
            x0(i) = x0(i) + dx;
            cvec<dim.nx> f;
            f.resize(dim.nx.num());
            fUser(f, x0, u0);
            x0(i) = x0(i) - dx;
            cvec<dim.nx> df;
            df = (f - f0) / dx;
            Jx.block(0, i, dim.nx.num(), 1) = df;
        }

        cvec<dim.nu> Ua = u0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < dim.nu.num(); i++) {
            Ua(i) = (Ua(i) < 1) ? 1 : Ua(i);
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < dim.nu.num(); i++) {
            // TODO support measured disturbaces
            int k = i;
            double du = dv * Ua(k);
            u0(k) = u0(k) + du;
            cvec<dim.nx> f;
            f.resize(dim.nx.num());
            fUser(f, x0, u0);
            u0(k) = u0(k) - du;
            cvec<dim.nx> df;
            df = (f - f0) / du;
            Jmv.block(0, i, dim.nx.num(), 1) = df;
        }
    }

    bool ctime;

    cvec<dim.ph * dim.nx> ceq;
    mat<(dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>(), dim.ph * dim.nx> Jceq;

    cvec<Dim<2>() * dim.ph * dim.ny> cineq;
    mat<(dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>(), Dim<2>() * dim.ph * dim.ny> Jcineq;

    cvec<dim.eq> ceq_user;
    mat<(dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>(), dim.eq> Jceq_user;

    cvec<dim.ineq> cineq_user;
    mat<(dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>(), dim.ineq> Jcineq_user;

    typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::StateFunHandle fUser = nullptr;
    typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle ieqUser = nullptr;
    typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle eqUser = nullptr;
    typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::OutFunHandle outUser = nullptr;

    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::mapping;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::x0;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::Xmat;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::Umat;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::e;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::ts;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::niteration;
};
} // namespace mpc
