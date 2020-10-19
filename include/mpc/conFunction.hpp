#pragma once

#include <mpc/baseFunction.hpp>
#include <mpc/mpc.hpp>

namespace mpc {
template <std::size_t Tnx, std::size_t Tnu, std::size_t Tny, std::size_t Tph, std::size_t Tch, std::size_t Tineq, std::size_t Teq>
class ConFunction : public BaseFunction<Tnx, Tnu, Tph, Tch> {
    using BaseFunction<Tnx, Tnu, Tph, Tch>::mapping;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::x0;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::Xmat;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::Umat;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::e;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::ts;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::ctime;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::niteration;

public:
    template <std::size_t Tcon>
    struct Cost {
        cvec<Tcon> value;
        cvec<Tcon * DecVarsSize> grad;
    };

    ConFunction()
        : nx(Tnx)
    {
    }
    ~ConFunction() = default;

    bool hasIneqConstraintFunction(void)
    {
        return ieqUser != nullptr;
    }

    bool hasEqConstraintFunction(void)
    {
        return eqUser != nullptr;
    }

    bool setStateSpaceFunction(const StateFunHandle<Tnx, Tnu> handle)
    {
        return fUser = handle, true;
    }

    bool setOutputFunction(const OutFunHandle handle)
    {
        return outUser = handle, true;
    }

    bool setIneqConstraintFunction(const IConFunHandle<Tineq, Tph, Tnx, Tnu> handle)
    {
        return ieqUser = handle, true;
    }

    bool setEqConstraintFunction(const EConFunHandle<Teq, Tph, Tnx, Tnu> handle)
    {
        return eqUser = handle, true;
    }

    Cost<StateIneqSize> evaluateIneq(cvec<DecVarsSize> x, bool hasGradient)
    {
        Cost<StateIneqSize> c;
        mapping.unwrapVector(x, x0, Xmat, Umat, e);

        // Set MPC constraints
        getStateIneqConstraints();

        c.value = cineq;
        c.grad = Eigen::Map<cvec<StateIneqSize * DecVarsSize>>(Jcineq.data(), Jcineq.size());

        dbg(Logger::DEEP) << "State inequality constraints value:\n"
                          << std::setprecision(10) << c.value << std::endl;
        if (!hasGradient) {
            dbg(Logger::DEEP) << "Gradient state inequality constraints not currectly used"
                              << std::endl;
        } else {
            dbg(Logger::DEEP) << "State inequality constraints gradient:\n"
                              << std::setprecision(10) << c.grad << std::endl;
        }

        return c;
    }

    Cost<Tineq> evaluateUserIneq(cvec<DecVarsSize> x, bool hasGradient)
    {
        Cost<Tineq> c;
        mapping.unwrapVector(x, x0, Xmat, Umat, e);

        // Add user defined constraints
        if (hasIneqConstraintFunction()) {
            cineq_user = ieqUser(Xmat, Umat, e);
            mat3<Tineq, Tnx, Tph> Jieqx;
            mat3<Tineq, Tnu, Tph> Jieqmv;
            cvec<Tineq> Jie;
            computeUserIneqJacobian(Jieqx, Jieqmv, Jie, Xmat, Umat, e, cineq_user);
            glueJacobian<Tineq>(Jcineq_user, Jieqx, Jieqmv, Jie);

            // TODO support for jacobian scaling
        } else {
            cineq_user.setZero();
            Jcineq_user.setZero();
        }

        c.value = cineq_user;
        c.grad = Eigen::Map<cvec<Tineq * DecVarsSize>>(Jcineq_user.data(), Jcineq_user.size());

        dbg(Logger::DEEP) << "User inequality constraints value:\n"
                          << std::setprecision(10) << c.value << std::endl;
        if (!hasGradient) {
            dbg(Logger::DEEP) << "Gradient user inequality constraints not currectly used"
                              << std::endl;
        } else {
            dbg(Logger::DEEP) << "User inequality constraints gradient:\n"
                              << std::setprecision(10) << c.grad << std::endl;
        }

        return c;
    }

    // cvec<getIneqSize()> ctot;
    // mat<PROB_SIZE, getIneqSize()> Jtot;

    // ctot.middleRows(0, 0, (2 * Tph * Tny), 1) = cineq;
    // ctot.middleRows((2 * Tph * Tny), 0, Tineq, 1) = cineq_user;
    // Jtot.middleCols(0, 0, PROB_SIZE, 2 * Tph * Tny) = Jcineq;
    // Jtot.middleCols(0, PROB_SIZE, 2 * Tph * Tny, Tineq) = Jcineq_user;

    Cost<StateEqSize> evaluateEq(cvec<DecVarsSize> x, bool hasGradient)
    {
        Cost<StateEqSize> c;
        dbg(Logger::DEEP) << "x: " << std::endl
                          << x << std::endl
                          << "-----------------" << std::endl;
        mapping.unwrapVector(x, x0, Xmat, Umat, e);
        dbg(Logger::DEEP) << "x0: " << std::endl
                          << x0 << std::endl
                          << "-----------------" << std::endl;
        dbg(Logger::DEEP) << "Xmat: " << std::endl
                          << Xmat << std::endl
                          << "-----------------" << std::endl;
        dbg(Logger::DEEP) << "Umat: " << std::endl
                          << Umat << std::endl
                          << "-----------------" << std::endl;

        // Set MPC constraints
        getStateEqConstraints(hasGradient);

        dbg(Logger::DEEP) << "ceq: " << std::endl
                          << ceq << std::endl
                          << "-----------------" << std::endl;

        c.value = ceq;
        c.grad = Eigen::Map<cvec<StateEqSize * DecVarsSize>>(Jceq.data(), Jceq.size());

        dbg(Logger::DEEP) << "State equality constraints value:\n"
                          << std::setprecision(10) << c.value << std::endl;
        if (!hasGradient) {
            dbg(Logger::DEEP) << "State equality constraints gradient not currectly used"
                              << std::endl;
        } else {
            dbg(Logger::DEEP) << "State equality constraints gradient:\n"
                              << std::setprecision(10) << c.grad << std::endl;
        }

        return c;
    }

    Cost<Teq> evaluateUserEq(cvec<DecVarsSize> x, bool hasGradient)
    {
        Cost<Teq> c;
        mapping.unwrapVector(x, x0, Xmat, Umat, e);

        // Add user defined constraints
        if (hasEqConstraintFunction()) {
            ceq_user = eqUser(Xmat, Umat);

            mat3<Teq, Tnx, Tph> Jeqx;
            mat3<Teq, Tnu, Tph> Jeqmv;
            computeUserEqJacobian(Jeqx, Jeqmv, Xmat, Umat, ceq_user);
            glueJacobian<Teq>(Jceq_user, Jeqx, Jeqmv, cvec<Teq>::Zero());

            // TODO support for jacobian scaling
        } else {
            ceq_user.setZero();
            Jceq_user.setZero();
        }

        c.value = ceq_user;
        c.grad = Eigen::Map<cvec<Tineq * DecVarsSize>>(Jceq_user.data(), Jceq_user.size());

        dbg(Logger::DEEP) << "User equality constraints value:\n"
                          << std::setprecision(10) << c.value << std::endl;
        if (!hasGradient) {
            dbg(Logger::DEEP) << "Gradient user equality constraints not currectly used"
                              << std::endl;
        } else {
            dbg(Logger::DEEP) << "User equality constraints gradient:\n"
                              << std::setprecision(10) << c.grad << std::endl;
        }

        return c;
    }

    // cvec<getEqSize()> ctot;
    // mat<PROB_SIZE, getEqSize()> Jtot;

    // ctot.middleRows(0, 0, (Tph * Tnx), 1) = ceq;
    // ctot.middleRows((Tph * Tnx), 0, Teq, 1) = ceq_user;
    // Jtot.middleCols(0, 0, PROB_SIZE, Tph * Tnx) = Jceq;
    // Jtot.middleCols(0, PROB_SIZE, Tph * Tnx, Teq) = Jceq;

    // c.value = ctot;
    // c.grad = Jtot.array();

private:
    template <std::size_t Tnc>
    void
    glueJacobian(mat<DecVarsSize, Tnc>& Jres, mat3<Tnc, Tnx, Tph> Jstate, mat3<Tnc, Tnu, Tph> Jmanvar, cvec<Tnc> Jcon)
    {
        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            Jres.middleRows(i * Tnx, Tnx) = Jstate[i].transpose();
        }

        mat<Tnc, Tph * Tnu> Jmanvar_mat;
        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            Jmanvar_mat.block(0, i * Tnu, Tnc, Tnu) = Jmanvar[i];
        }

        Jres.middleRows(Tph * Tnx, Tch * Tnu) = (Jmanvar_mat * mapping.Iz2u).transpose();
        Jres.bottomRows(1) = Jcon.transpose();
    }

    void getStateIneqConstraints()
    {
        // TODO manage output bounds
        return;
    }

    void getStateEqConstraints(bool hasGradient)
    {
        ceq.setZero();
        Jceq.setZero();

        mat3<Tph * Tnx, Tnx, Tph> Jx;
        //#pragma omp parallel for
        for (size_t i = 0; i < Jx.size(); i++) {
            Jx[i].setZero();
        }

        // TODO support measured noise
        mat3<Tph * Tnx, Tnu, Tph> Jmv;
        //#pragma omp parallel for
        for (size_t i = 0; i < Jmv.size(); i++) {
            Jmv[i].setZero();
        }

        cvec<Tph * Tnx> Je;
        Je.setZero();

        int ic = 0;

        mat<Tnx, Tnx> Ix;
        Ix.setIdentity(Tnx, Tnx);

        // TODO support scaling
        mat<Tnx, Tnx> Sx, Tx;
        Sx.setIdentity(Tnx, Tnx);
        Tx.setIdentity(Tnx, Tnx);

        // TODO bind for continuos time
        if (ctime) {
            //#pragma omp parallel for
            for (size_t i = 0; i < Tph; i++) {
                cvec<Tnu> uk = Umat.row(i).transpose();
                cvec<Tnx> xk = Xmat.row(i).transpose();

                double h = ts / 2.0;
                cvec<Tnx> xk1 = Xmat.row(i + 1).transpose();
                cvec<Tnx> fk = fUser(xk, uk);
                cvec<Tnx> fk1 = fUser(xk1, uk);
                ceq.middleRows(ic, Tnx) = xk + (h * (fk + fk1)) - xk1;
                // TODO support scaling
                ceq.middleRows(ic, Tnx) = ceq.middleRows(ic, Tnx) / 1.0;

                if (hasGradient) {
                    mat<Tnx, Tnx> Ak;
                    mat<Tnx, Tnu> Bk;
                    computeStateEqJacobian(Ak, Bk, fk, xk, uk);
                    mat<Tnx, Tnx> Ak1;
                    mat<Tnx, Tnu> Bk1;
                    computeStateEqJacobian(Ak1, Bk1, fk1, xk1, uk);

                    if (i > 0) {
                        Jx[i - 1].middleRows(ic, Tnx) = Ix + (h * Sx * Ak * Tx);
                    }
                    Jx[i].middleRows(ic, Tnx) = -Ix + (h * Sx * Ak1 * Tx);
                    Jmv[i].middleRows(ic, Tnx) = h * Sx * (Bk + Bk1);
                }

                ic += Tnx;
            }
        } else {
            //#pragma omp parallel for
            for (size_t i = 0; i < Tph; i++) {
                cvec<Tnu> uk = Umat.row(i).transpose();
                cvec<Tnx> xk = Xmat.row(i).transpose();
                cvec<Tnx> xk1 = fUser(xk, uk);
                ceq.middleRows(ic, Tnx) = Xmat.row(i + 1).transpose() - xk1;
                // TODO support scaling
                ceq.middleRows(ic, Tnx) = ceq.middleRows(ic, Tnx) / 1.0;

                if (hasGradient) {
                    mat<Tnx, Tnx> Ak;
                    mat<Tnx, Tnu> Bk;
                    computeStateEqJacobian(Ak, Bk, xk1, xk, uk);

                    Ak = Sx * Ak * Tx;
                    Bk = Sx * Bk;

                    Jx[i].middleRows(ic, Tnx) = Ix;
                    if (i > 0) {
                        Jx[i - 1].middleRows(ic, Tnx) = -Ak;
                    }
                    Jmv[i].middleRows(ic, Tnx) = -Bk;
                }

                ic += Tnx;
            }
        }

        if (hasGradient) {
            glueJacobian<Tph * Tnx>(Jceq, Jx, Jmv, Je);
        }
    }

    void computeUserIneqJacobian(mat3<Tineq, Tnx, Tph>& Jconx, mat3<Tineq, Tnu, Tph>& Jconmv, cvec<Tineq>& Jcone, mat<Tph + 1, Tnx> x0, mat<Tph + 1, Tnu> u0, double e0, cvec<Tineq> f0)
    {
        double dv = 1e-6;

        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            Jconx[i].setZero();
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            Jconmv[i].setZero();
        }

        Jcone.setZero();

        mat<Tph + 1, Tnx> Xa = x0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < (size_t)Xa.rows(); i++) {
            for (size_t j = 0; j < (size_t)Xa.cols(); j++) {
                Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            for (size_t j = 0; j < Tnx; j++) {
                int ix = i + 1;
                double dx = dv * Xa.array()(j);
                x0(ix, j) = x0(ix, j) + dx;
                cvec<Tineq> f = ieqUser(x0, u0, e);
                x0(ix, j) = x0(ix, j) - dx;
                cvec<Tineq> df = (f - f0) / dx;
                Jconx[i].col(j) = df;
            }
        }

        mat<Tph + 1, Tnu> Ua = u0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < (size_t)Ua.rows(); i++) {
            for (size_t j = 0; j < (size_t)Ua.cols(); j++) {
                Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < Tph - 1; i++) {
            // TODO support measured disturbaces
            for (size_t j = 0; j < Tnu; j++) {
                int k = j;
                double du = dv * Ua.array()(k);
                u0(i, k) = u0(i, k) + du;
                cvec<Tineq> f = ieqUser(x0, u0, e);
                u0(i, k) = u0(i, k) - du;
                cvec<Tineq> df = (f - f0) / du;
                Jconmv[i].col(j) = df;
            }
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t j = 0; j < Tnu; j++) {
            int k = j;
            double du = dv * Ua.array()(k);
            u0(Tph - 1, k) = u0(Tph - 1, k) + du;
            u0(Tph, k) = u0(Tph, k) + du;
            cvec<Tineq> f = ieqUser(x0, u0, e);
            u0(Tph - 1, k) = u0(Tph - 1, k) - du;
            u0(Tph, k) = u0(Tph, k) - du;
            cvec<Tineq> df = (f - f0) / du;
            Jconmv[Tph - 1].col(j) = df;
        }

        double ea = fmax(1e-6, abs(e0));
        double de = ea * dv;
        cvec<Tineq> f1 = ieqUser(x0, u0, e0 + de);
        cvec<Tineq> f2 = ieqUser(x0, u0, e0 - de);
        Jcone = (f1 - f2) / (2 * de);
    }

    void computeUserEqJacobian(mat3<Teq, Tnx, Tph>& Jconx, mat3<Teq, Tnu, Tph>& Jconmv, mat<Tph + 1, Tnx> x0, mat<Tph + 1, Tnu> u0, cvec<Teq> f0)
    {
        double dv = 1e-6;

        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            Jconx[i].setZero();
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            Jconmv[i].setZero();
        }

        mat<Tph + 1, Tnx> Xa = x0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < (size_t)Xa.rows(); i++) {
            for (size_t j = 0; j < (size_t)Xa.cols(); j++) {
                Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < Tph; i++) {
            for (size_t j = 0; j < Tnx; j++) {
                int ix = i + 1;
                double dx = dv * Xa.array()(j);
                x0(ix, j) = x0(ix, j) + dx;
                cvec<Teq> f = eqUser(x0, u0);
                x0(ix, j) = x0(ix, j) - dx;
                cvec<Teq> df = (f - f0) / dx;
                Jconx[i].col(j) = df;
            }
        }

        mat<Tph + 1, Tnu> Ua = u0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < (size_t)Ua.rows(); i++) {
            for (size_t j = 0; j < (size_t)Ua.cols(); j++) {
                Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < Tph - 1; i++) {
            // TODO support measured disturbaces
            for (size_t j = 0; j < Tnu; j++) {
                int k = j;
                double du = dv * Ua.array()(k);
                u0(i, k) = u0(i, k) + du;
                cvec<Teq> f = eqUser(x0, u0);
                u0(i, k) = u0(i, k) - du;
                cvec<Teq> df = (f - f0) / du;
                Jconmv[i].col(j) = df;
            }
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t j = 0; j < Tnu; j++) {
            int k = j;
            double du = dv * Ua.array()(k);
            u0(Tph - 1, k) = u0(Tph - 1, k) + du;
            u0(Tph, k) = u0(Tph, k) + du;
            cvec<Teq> f = eqUser(x0, u0);
            u0(Tph - 1, k) = u0(Tph - 1, k) - du;
            u0(Tph, k) = u0(Tph, k) - du;
            cvec<Teq> df = (f - f0) / du;
            Jconmv[Tph - 1].col(j) = df;
        }
    }

    void computeStateEqJacobian(mat<Tnx, Tnx>& Jx, mat<Tnx, Tnu>& Jmv, cvec<Tnx> f0, cvec<Tnx> x0, cvec<Tnu> u0)
    {
        Jx.setZero();
        Jmv.setZero();

        double dv = 1 - 6;

        cvec<Tnx> Xa = x0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < Tnx; i++) {
            Xa(i) = (Xa(i) < 1) ? 1 : Xa(i);
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < Tnx; i++) {
            double dx = dv * Xa(i);
            x0(i) = x0(i) + dx;
            cvec<Tnx> f = fUser(x0, u0);
            x0(i) = x0(i) - dx;
            cvec<Tnx> df = (f - f0) / dx;
            Jx.block(0, i, Tnx, 1) = df;
        }

        cvec<Tnu> Ua = u0.cwiseAbs();
        //#pragma omp parallel for
        for (size_t i = 0; i < Tnu; i++) {
            Ua(i) = (Ua(i) < 1) ? 1 : Ua(i);
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < Tnu; i++) {
            // TODO support measured disturbaces
            int k = i;
            double du = dv * Ua(k);
            u0(k) = u0(k) + du;
            cvec<Tnx> f = fUser(x0, u0);
            u0(k) = u0(k) - du;
            cvec<Tnx> df = (f - f0) / du;
            Jmv.block(0, i, Tnx, 1) = df;
        }
    }

    int nx;

    cvec<Tph * Tnx> ceq;
    mat<DecVarsSize, Tph * Tnx> Jceq;

    cvec<2 * Tph * Tny> cineq;
    mat<DecVarsSize, 2 * Tph * Tny> Jcineq;

    cvec<Teq> ceq_user;
    mat<DecVarsSize, Teq> Jceq_user;

    cvec<Tineq> cineq_user;
    mat<DecVarsSize, Tineq> Jcineq_user;

    StateFunHandle<Tnx, Tnu> fUser = nullptr;
    IConFunHandle<Tineq, Tph, Tnx, Tnu> ieqUser = nullptr;
    EConFunHandle<Teq, Tph, Tnx, Tnu> eqUser = nullptr;
    OutFunHandle outUser = nullptr;
};
} // namespace mpc