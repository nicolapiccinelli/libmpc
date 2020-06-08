#pragma once

#include <mpc/mapping.hpp>
#include <mpc/mpc.hpp>
#include <mpc/conFunction.hpp>
#include <mpc/objFunction.hpp>
#include <mpc/logger.hpp>
#include <nlopt.hpp>

namespace mpc {
template <std::size_t Tnx, std::size_t Tnu, std::size_t Tny, std::size_t Tph, std::size_t Tch, std::size_t Tineq, std::size_t Teq>
class Optimizer {
public:
    Optimizer(bool hardConstraints)
        : hard(hardConstraints)
    {
        inner_opt = new nlopt::opt(nlopt::LD_SLSQP, DecVarsSize);
        setDefaultBounds();

        Parameters p;
        p.relative_ftol = 1e-10;
        p.maximum_iteration = 100;
        p.relative_xtol = 1e-10;
        setTolerances(p);

        last_r.cmd.setZero();
    }

    ~Optimizer()
    {
        delete inner_opt;
    }

    void setMapping(Common<Tnx, Tnu, Tph, Tch>& m)
    {
        mapping = m;
    }

    void setDefaultBounds()
    {
        std::vector<double> lb, ub;
        lb.resize(DecVarsSize);
        ub.resize(DecVarsSize);

        for (size_t i = 0; i < DecVarsSize; i++) {
            lb[i] = -std::numeric_limits<double>::infinity();
            if (i + 1 == DecVarsSize && hard) {
                lb[i] = 0;
            }
            ub[i] = std::numeric_limits<double>::infinity();
        }

        inner_opt->set_lower_bounds(lb);
        inner_opt->set_upper_bounds(ub);

        // TODO support output bounds
        outputBounds = false;
    }

    void setTolerances(Parameters param)
    {
        inner_opt->set_ftol_rel(param.relative_ftol);
        inner_opt->set_maxeval(param.maximum_iteration);
        inner_opt->set_xtol_rel(param.relative_xtol);

        Logger::instance().log(Logger::log_type::DEBUG) << "Setting tolerances and stopping criterias" << std::endl;
    }

    bool bind(ObjFunction<Tnx, Tnu, Tph, Tch>* objFunc)
    {
        try {
            inner_opt->set_min_objective(Optimizer::nloptObjFunWrapper, objFunc);
            return true;
        } catch (const std::exception& e) {
            Logger::instance().log(Logger::log_type::DEBUG) << "Unable to bind objective function: "
                              << e.what() << std::endl;
            return false;
        }
    }

    bool bindIneq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<StateIneqSize> tol)
    {
        try {
            if (outputBounds) {
                inner_opt->add_inequality_mconstraint(
                            Optimizer::nloptIneqConFunWrapper,
                            conFunc,
                            std::vector<double>(
                                tol.data(),
                                tol.data() + tol.rows() * tol.cols()));
                Logger::instance().log(Logger::log_type::DEBUG) << "Adding state defined inequality constraints" << std::endl;
            } else {
                Logger::instance().log(Logger::log_type::DEBUG) << "State inequality constraints skipped" << std::endl;
            }
            return true;
        } catch (const std::exception& e) {
            Logger::instance().log(Logger::log_type::DEBUG) << "Unable to bind constraints function\n"
                              << e.what() << '\n';
            return false;
        }
    }

    bool bindEq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<StateEqSize> tol)
    {
        try {
            inner_opt->add_equality_mconstraint(
                        Optimizer::nloptEqConFunWrapper,
                        conFunc,
                        std::vector<double>(
                            tol.data(),
                            tol.data() + tol.rows() * tol.cols()));
            Logger::instance().log(Logger::log_type::DEBUG) << "Adding state defined equality constraints" << std::endl;
            return true;
        } catch (const std::exception& e) {
            Logger::instance().log(Logger::log_type::DEBUG) << "Unable to bind constraints function\n"
                              << e.what() << '\n';
            return false;
        }
    }

    bool bindUserIneq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<Tineq> tol)
    {
        try {
            inner_opt->add_inequality_mconstraint(
                        Optimizer::nloptUserIneqConFunWrapper,
                        conFunc,
                        std::vector<double>(
                            tol.data(),
                            tol.data() + tol.rows() * tol.cols()));
            Logger::instance().log(Logger::log_type::DEBUG) << "Adding user inequality constraints" << std::endl;
            return true;
        } catch (const std::exception& e) {
            Logger::instance().log(Logger::log_type::DEBUG) << "Unable to bind constraints function\n"
                              << e.what() << '\n';
            return false;
        }
    }

    bool bindUserEq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<Teq> tol)
    {
        try {
            inner_opt->add_equality_mconstraint(
                        Optimizer::nloptUserEqConFunWrapper,
                        conFunc,
                        std::vector<double>(
                            tol.data(),
                            tol.data() + tol.rows() * tol.cols()));
            Logger::instance().log(Logger::log_type::DEBUG) << "Adding user equality constraints" << std::endl;
            return true;
        } catch (const std::exception& e) {
            Logger::instance().log(Logger::log_type::DEBUG) << "Unable to bind constraints function\n"
                              << e.what() << '\n';
            return false;
        }
    }

    Result<Tnu> run(
            const cvec<Tnx> x0,
            const cvec<Tnu> u0)
    {
        Result<Tnu> r;

        std::vector<double> optX0;
        optX0.resize(DecVarsSize);

        int counter = 0;
        for (size_t i = 0; i < Tph; i++) {
            for (size_t j = 0; j < Tnx; j++) {
                optX0[counter++] = x0[j];
            }
        }

        mat<Tph, Tnu> Umv;
        Umv.setZero();
        for (size_t i = 0; i < Tph; i++) {
            for (size_t j = 0; j < Tnu; j++) {
                Umv(i, j) = u0[j];
            }
        }

        cvec<Tph * Tnu> UmvVectorized;
        int vec_counter = 0;
        for (size_t i = 0; i < Tph; i++) {
            for (size_t j = 0; j < Tnu; j++) {
                UmvVectorized[vec_counter++] = Umv(i, j);
            }
        }

        cvec<Tch* Tnu> res = mapping.Iu2z * UmvVectorized;
        for (size_t j = 0; j < Tch * Tnu; j++) {
            optX0[counter++] = res[j];
        }

        optX0[DecVarsSize - 1] = curr_slack;

        try {
            cvec<DecVarsSize> opt_vector(inner_opt->optimize(optX0).data());
            r.cost = inner_opt->last_optimum_value();
            r.retcode = inner_opt->last_optimize_result();

            Logger::instance().log(Logger::log_type::INFO) << "Optimization end after: "
                              << inner_opt->get_numevals()
                              << " evaluation steps" << std::endl;
            Logger::instance().log(Logger::log_type::INFO) << "Optimization end with code: "
                              << r.retcode << std::endl;
            Logger::instance().log(Logger::log_type::INFO) << "Optimization end with cost: "
                              << r.cost << std::endl;

            mat<Tph + 1, Tnx> Xmat;
            mat<Tph + 1, Tnu> Umat;

            mapping.unwrapVector(opt_vector, x0, Xmat, Umat, curr_slack);

            Logger::instance().log(Logger::log_type::DEBUG) << "Optimal predicted state vector\n"
                              << Xmat << std::endl;
            Logger::instance().log(Logger::log_type::DEBUG) << "Optimal predicted output vector\n"
                              << Umat << std::endl;
            r.cmd = Umat.row(0);
        } catch (const std::exception& e) {
            Logger::instance().log(Logger::log_type::INFO) << "No optimal solution found: " << e.what() << std::endl;
            r.cmd = last_r.cmd;
            r.retcode = -1;
        }

        last_r = r;
        return r;
    }

private:
    nlopt::opt* inner_opt;
    Result<Tnu> last_r;
    double curr_slack;
    bool hard;
    bool outputBounds;

    Common<Tnx, Tnu, Tph, Tch> mapping;

    static double nloptObjFunWrapper(const std::vector<double>& x, std::vector<double>& grad, void* objFunc)
    {
        bool hasGradient = !grad.empty();

        cvec<DecVarsSize> x_arr(x.data());
        auto res = ((ObjFunction<Tnx, Tnu, Tph, Tch>*)objFunc)->evaluate(x_arr, hasGradient);
        if (hasGradient) {
            // The gradient should be transposed since the difference between matlab and nlopt
            std::copy_n(&res.grad.transpose()[0], res.grad.transpose().cols() * res.grad.transpose().rows(), grad.begin());
        }
        return res.value;
    }

    static void nloptIneqConFunWrapper(unsigned int m, double* result, unsigned int n, const double* x, double* grad, void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        cvec<DecVarsSize> x_arr(x);
        ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* tmp = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc);

        typename ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::template Cost<StateIneqSize> res = tmp->evaluateIneq(x_arr, hasGradient);
        std::memcpy(result, res.value.data(), StateIneqSize * sizeof(double));

        if (hasGradient) {
            // The gradient should be transposed since the difference between matlab and nlopt
            std::memcpy(grad, res.grad.transpose().data(), StateIneqSize * DecVarsSize * sizeof(double));
        }
    }

    static void nloptEqConFunWrapper(unsigned int m, double* result, unsigned int n, const double* x, double* grad, void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        cvec<DecVarsSize> x_arr(x);
        ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* tmp = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc);

        typename ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::template Cost<StateEqSize> res = tmp->evaluateEq(x_arr, hasGradient);
        std::memcpy(result, res.value.data(), StateEqSize * sizeof(double));

        if (hasGradient) {
            // The gradient should be transposed since the difference between matlab and nlopt
            std::memcpy(grad, res.grad.transpose().data(), StateEqSize * DecVarsSize * sizeof(double));
        }
    }

    static void nloptUserIneqConFunWrapper(unsigned int m, double* result, unsigned int n, const double* x, double* grad, void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        cvec<DecVarsSize> x_arr(x);
        ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* tmp = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc);

        typename ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::template Cost<Tineq> res = tmp->evaluateUserIneq(x_arr, hasGradient);
        std::memcpy(result, res.value.data(), Tineq * sizeof(double));

        if (hasGradient) {
            // The gradient should be transposed since the difference between matlab and nlopt
            std::memcpy(grad, res.grad.transpose().data(), Tineq * DecVarsSize * sizeof(double));
        }
    }

    static void nloptUserEqConFunWrapper(unsigned int m, double* result, unsigned int n, const double* x, double* grad, void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        cvec<DecVarsSize> x_arr(x);
        ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* tmp = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc);

        typename ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::template Cost<Teq> res = tmp->evaluateUserEq(x_arr, hasGradient);
        std::memcpy(result, res.value.data(), Teq * sizeof(double));

        if (hasGradient) {
            // The gradient should be transposed since the difference between matlab and nlopt
            std::memcpy(grad, res.grad.transpose().data(), Teq * DecVarsSize * sizeof(double));
        }
    }
};
} // namespace mpc
