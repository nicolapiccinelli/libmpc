#pragma once

#include <mpc/Base.hpp>

namespace mpc {

/**
 * @brief Managment of the objective function for the non-linear mpc
 * 
 * @tparam Tnx dimension of the state space
 * @tparam Tnu dimension of the input space
 * @tparam Tny dimension of the output space
 * @tparam Tph length of the prediction horizon
 * @tparam Tch length of the control horizon
 * @tparam Tineq number of the user inequality constraints
 * @tparam Teq number of the user equality constraints
 */
template <
    int Tnx, int Tnu, int Tny,
    int Tph, int Tch,
    int Tineq, int Teq>
class Objective : public Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq> {
private:
    using Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::checkOrQuit;
    using Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::dim;

public:

    /**
     * @brief Internal structure containing the value and the gradient
     * of the evaluated constraints.
     */
    struct Cost {
        double value;
        cvec<((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>())> grad;
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
        x0.resize(dim.nx.num());
        Xmat.resize((dim.ph.num() + 1), dim.nx.num());
        Umat.resize((dim.ph.num() + 1), dim.nu.num());
        Jx.resize(dim.nx.num(), dim.ph.num());
        Jmv.resize(dim.nu.num(), dim.ph.num());

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
        const typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::ObjFunHandle handle)
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
        cvec<((dim.ph * dim.nx) + (dim.nu * dim.ch) + Dim<1>())> x,
        bool hasGradient)
    {
        checkOrQuit();

        Cost c;
        c.grad.resize(((dim.ph.num() * dim.nx.num()) + (dim.nu.num() * dim.ch.num()) +1));

        mapping.unwrapVector(x, x0, Xmat, Umat, e);
        c.value = fuser(Xmat, Umat, e);

        if (hasGradient) {
            computeJacobian(Xmat, Umat, c.value, e);

            for (int i = 0; i < Xmat.cols(); i++) {
                Xmat.col(i) /= 1.0 / mapping.StateScaling()(i);
            }

            int counter = 0;

            //#pragma omp parallel for
            for (int j = 0; j < (int)Jx.cols(); j++) {
                for (int i = 0; i < (int)Jx.rows(); i++) {
                    c.grad(counter++) = Jx(i, j);
                }
            }

            cvec<(dim.ph * dim.nu)> JmvVectorized;
            JmvVectorized.resize((dim.ph.num() * dim.nu.num()));

            int vec_counter = 0;
            //#pragma omp parallel for
            for (int j = 0; j < (int)Jmv.cols(); j++) {
                for (int i = 0; i < (int)Jmv.rows(); i++) {
                    JmvVectorized(vec_counter++) = Jmv(i, j);
                }
            }

            cvec<(dim.nu * dim.ch)> res;
            res = mapping.Iz2u().transpose() * JmvVectorized;
            //#pragma omp parallel for
            for (size_t j = 0; j < (dim.ch.num() * dim.nu.num()); j++) {
                c.grad(counter++) = res(j);
            }

            c.grad(((dim.ph.num() * dim.nx.num()) + (dim.nu.num() * dim.ch.num()) + 1) - 1) = Je;
        }

        Logger::instance().log(Logger::log_type::DETAIL)
            << "("
            << niteration
            << ") Objective function value: \n"
            << std::setprecision(10)
            << c.value
            << std::endl;
        if (!hasGradient) {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "("
                << niteration
                << ") Gradient not currectly used"
                << std::endl;
        } else {
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
        mat<(dim.ph + Dim<1>()), Tnx> x0,
        mat<(dim.ph + Dim<1>()), Tnu> u0,
        double f0,
        double e0)
    {
        double dv = 1e-6;

        Jx.setZero();
        // TODO support measured disturbaces
        Jmv.setZero();

        mat<(dim.ph + Dim<1>()), Tnx> Xa;
        Xa = x0.cwiseAbs();

        //#pragma omp parallel for
        for (int i = 0; i < Xa.rows(); i++) {
            for (int j = 0; j < Xa.cols(); j++) {
                Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
            }
        }

        //#pragma omp parallel for
        for (size_t i = 0; i < dim.ph.num(); i++) {
            for (size_t j = 0; j < dim.nx.num(); j++) {
                int ix = i + 1;
                double dx = dv * Xa.array()(j);
                x0(ix, j) = x0(ix, j) + dx;
                double f = fuser(x0, u0, e0);
                x0(ix, j) = x0(ix, j) - dx;
                double df = (f - f0) / dx;
                Jx(j, i) = df;
            }
        }

        mat<(dim.ph + Dim<1>()), dim.nu> Ua;
        Ua = u0.cwiseAbs();

        //#pragma omp parallel for
        for (int i = 0; i < Ua.rows(); i++) {
            for (int j = 0; j < Ua.cols(); j++) {
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
                double f = fuser(x0, u0, e0);
                u0(i, k) = u0(i, k) - du;
                double df = (f - f0) / du;
                Jmv(j, i) = df;
            }
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (size_t j = 0; j < dim.nu.num(); j++) {
            int k = j;
            double du = dv * Ua.array()(k);
            u0((dim.ph.num() - 1), k) = u0((dim.ph.num() - 1), k) + du;
            u0(dim.ph.num(), k) = u0(dim.ph.num(), k) + du;
            double f = fuser(x0, u0, e0);
            u0((dim.ph.num() - 1), k) = u0((dim.ph.num() - 1), k) - du;
            u0(dim.ph.num(), k) = u0(dim.ph.num(), k) - du;
            double df = (f - f0) / du;
            Jmv(j, (dim.ph.num() - 1)) = df;
        }

        double ea = fmax(1e-6, abs(e0));
        double de = ea * dv;
        double f1 = fuser(x0, u0, e0 + de);
        double f2 = fuser(x0, u0, e0 - de);
        Je = (f1 - f2) / (2 * de);
    }

    typename Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::ObjFunHandle fuser = nullptr;

    mat<dim.nx, dim.ph> Jx;
    mat<dim.nu, dim.ph> Jmv;

    double Je;

    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::mapping;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::x0;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::Xmat;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::Umat;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::e;
    using Base<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::niteration;
};
} // namespace mpc
