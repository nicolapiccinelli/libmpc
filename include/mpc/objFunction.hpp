#pragma once

#include <mpc/baseFunction.hpp>
#include <mpc/mpc.hpp>

namespace mpc {
template <std::size_t Tnx, std::size_t Tnu, std::size_t Tph, std::size_t Tch>
class ObjFunction : public BaseFunction<Tnx, Tnu, Tph, Tch> {

    using BaseFunction<Tnx, Tnu, Tph, Tch>::mapping;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::x0;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::Xmat;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::Umat;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::e;
    using BaseFunction<Tnx, Tnu, Tph, Tch>::niteration;

public:
    struct Cost {
        double value;
        cvec<DecVarsSize> grad;
    };

    ObjFunction() = default;
    ~ObjFunction() = default;

    bool setUserFunction(const ObjFunHandle<Tph, Tnx, Tnu> handle)
    {
        return fuser = handle, true;
    }

    Cost evaluate(cvec<DecVarsSize> x, bool hasGradient)
    {
        Cost c;
        mapping.unwrapVector(x, x0, Xmat, Umat, e);
        c.value = fuser(Xmat, Umat, e);
        if (hasGradient) {
            computeJacobian(Xmat, Umat, c.value, e);

            int counter = 0;
            //#pragma omp parallel for
            for (size_t j = 0; j < (size_t)Jx.cols(); j++) {
                for (size_t i = 0; i < (size_t)Jx.rows(); i++) {
                    c.grad[counter++] = Jx(i, j);
                }
            }

            cvec<Tph * Tnu> JmvVectorized;
            int vec_counter = 0;
            //#pragma omp parallel for
            for (size_t j = 0; j < (size_t)Jmv.cols(); j++) {
                for (size_t i = 0; i < (size_t)Jmv.rows(); i++) {
                    JmvVectorized[vec_counter++] = Jmv(i, j);
                }
            }

            cvec<Tch* Tnu> res = mapping.Iz2u.transpose() * JmvVectorized;
            //#pragma omp parallel for
            for (size_t j = 0; j < Tch * Tnu; j++) {
                c.grad[counter++] = res[j];
            }

            c.grad[DecVarsSize - 1] = Je;
        }

        // TODO support scaling

        dbg(Logger::DEEP) << "(" << niteration << ") Objective function value: \n"
                          << std::setprecision(10) << c.value << std::endl;
        if (!hasGradient) {
            dbg(Logger::DEEP) << "(" << niteration << ") Gradient not currectly used"
                              << std::endl;
        } else {
            dbg(Logger::DEEP) << "(" << niteration << ") Objective function gradient: \n"
                              << std::setprecision(10) << c.grad << std::endl;
        }

        // debug information
        niteration++;

        return c;
    }

private:
    void computeJacobian(mat<Tph + 1, Tnx> x0, mat<Tph + 1, Tnu> u0, double f0, double e0)
    {
        double dv = 1e-6;

        Jx.setZero();
        // TODO support measured disturbaces
        Jmv.setZero();

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
                double dx = dv * Xa(j, 0);
                x0(ix, j) = x0(ix, j) + dx;
                double f = fuser(x0, u0, e0);
                x0(ix, j) = x0(ix, j) - dx;
                double df = (f - f0) / dx;
                Jx(j, i) = df;
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
                double du = dv * Ua(k, 0);
                u0(i, k) = u0(i, k) + du;
                double f = fuser(x0, u0, e0);
                u0(i, k) = u0(i, k) - du;
                double df = (f - f0) / du;
                Jmv(j, i) = df;
            }
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t j = 0; j < Tnu; j++) {
            int k = j;
            double du = dv * Ua(k, 0);
            u0(Tph - 1, k) = u0(Tph - 1, k) + du;
            u0(Tph, k) = u0(Tph, k) + du;
            double f = fuser(x0, u0, e0);
            u0(Tph - 1, k) = u0(Tph - 1, k) - du;
            u0(Tph, k) = u0(Tph, k) - du;
            double df = (f - f0) / du;
            Jmv(j, Tph - 1) = df;
        }

        double ea = fmax(1e-6, abs(e0));
        double de = ea * dv;
        double f1 = fuser(x0, u0, e0 + de);
        double f2 = fuser(x0, u0, e0 - de);
        Je = (f1 - f2) / (2 * de);
    }

    ObjFunHandle<Tph, Tnx, Tnu> fuser = nullptr;

    mat<Tnx, Tph> Jx;
    mat<Tnu, Tph> Jmv;
    double Je;
};
} // namespace mpc
