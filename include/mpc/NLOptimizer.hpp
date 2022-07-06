#pragma once

#include <mpc/Constraints.hpp>
#include <mpc/IOptimizer.hpp>
#include <mpc/Logger.hpp>
#include <mpc/Mapping.hpp>
#include <mpc/Objective.hpp>
#include <mpc/Types.hpp>

#include <nlopt.hpp>

namespace mpc
{
    /**
     * @brief Non-lnear MPC optimizer interface class
     *
     * @tparam sizer.nx dimension of the state space
     * @tparam sizer.nu dimension of the input space
     * @tparam Tny dimension of the output space
     * @tparam Tph length of the prediction horizon
     * @tparam Tch length of the control horizon
     * @tparam Tineq number of the user inequality constraints
     * @tparam sizer.eq number of the user equality constraints
     */
    template <MPCSize sizer>
    class NLOptimizer : public IOptimizer<sizer>
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
        NLOptimizer() = default;

        ~NLOptimizer()
        {
            checkOrQuit();
            delete innerOpt;
        }

        /**
         * @brief Initialization hook override. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        void onInit()
        {
            innerOpt = new nlopt::opt(nlopt::LD_SLSQP, ((ph() * nx()) + (nu() * ch()) + 1));

            setParameters(NLParameters());

            last_r.cmd.resize(nu());
            last_r.cmd.setZero();

            currentSlack = 0;
        }

        /**
         * @brief Set the mapping object
         *
         * @param m mapping reference
         */
        void setMapping(Mapping<sizer> &m)
        {
            checkOrQuit();

            mapping = m;
        }

        /**
         * @brief Set the optmiziation parameters
         *
         * @param param parameters desired
         */
        void setParameters(const Parameters &param)
        {
            checkOrQuit();

            auto nl_param = dynamic_cast<const NLParameters *>(&param);

            innerOpt->set_ftol_rel(nl_param->relative_ftol);
            innerOpt->set_maxeval(nl_param->maximum_iteration);
            innerOpt->set_xtol_rel(nl_param->relative_xtol);

            std::vector<double> lb, ub;
            lb.resize(((ph() * nx()) + (nu() * ch()) + 1));
            ub.resize(((ph() * nx()) + (nu() * ch()) + 1));

            for (size_t i = 0; i < ((ph() * nx()) + (nu() * ch()) + 1); i++)
            {
                lb[i] = -std::numeric_limits<double>::infinity();
                if (i + 1 == ((ph() * nx()) + (nu() * ch()) + 1) && nl_param->hard_constraints)
                {
                    lb[i] = 0;
                }
                ub[i] = std::numeric_limits<double>::infinity();
            }

            innerOpt->set_lower_bounds(lb);
            innerOpt->set_upper_bounds(ub);

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting tolerances and stopping criterias"
                << std::endl;
        }

        /**
         * @brief Bind the objective function class with the internal solver
         * objective function referemce
         *
         * @param objFunc objective function class instance
         * @return true
         * @return false
         */
        bool bind(Objective<sizer> *objFunc)
        {
            checkOrQuit();

            try
            {
                innerOpt->set_min_objective(NLOptimizer::nloptObjFunWrapper, objFunc);
                return true;
            }
            catch (const std::exception &e)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Unable to bind objective function: "
                    << e.what()
                    << std::endl;
                return false;
            }
        }

        /**
         * @brief Bind the constraints class with the internal solver
         * system's dynamics equality constraints function referemce
         *
         * @param conFunc constraints class instance
         * @param tol equality constraints tolerances
         * @return true
         * @return false
         */
        bool bindEq(
            Constraints<sizer> *conFunc,
            constraints_type,
            const cvec<(sizer.ph * sizer.nx)> tol)
        {
            checkOrQuit();

            try
            {
                innerOpt->add_equality_mconstraint(
                    NLOptimizer::nloptEqConFunWrapper,
                    conFunc,
                    std::vector<double>(
                        tol.data(),
                        tol.data() + tol.rows() * tol.cols()));
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Adding state defined equality constraints"
                    << std::endl;
                return true;
            }
            catch (const std::exception &e)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Unable to bind constraints function\n"
                    << e.what()
                    << '\n';
                return false;
            }
        }

        /**
         * @brief Bind the constraints class with the internal solver
         * user inequality constraints function referemce
         *
         * @param conFunc constraints class instance
         * @param tol inequality constraints tolerances
         * @return true
         * @return false
         */
        bool bindUserIneq(
            Constraints<sizer> *conFunc,
            constraints_type,
            const cvec<sizer.ineq> tol)
        {
            checkOrQuit();

            try
            {
                innerOpt->add_inequality_mconstraint(
                    NLOptimizer::nloptUserIneqConFunWrapper,
                    conFunc,
                    std::vector<double>(
                        tol.data(),
                        tol.data() + tol.rows() * tol.cols()));
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Adding user inequality constraints"
                    << std::endl;
                return true;
            }
            catch (const std::exception &e)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Unable to bind constraints function\n"
                    << e.what()
                    << '\n';
                return false;
            }
        }

        /**
         * @brief Bind the constraints class with the internal solver
         * user equality constraints function referemce
         *
         * @param conFunc constraints class instance
         * @param type constraints type
         * @param tol equality constraints tolerances
         * @return true
         * @return false
         */
        bool bindUserEq(
            Constraints<sizer> *conFunc,
            constraints_type /*type*/,
            const cvec<sizer.eq> tol)
        {
            checkOrQuit();

            try
            {
                innerOpt->add_equality_mconstraint(
                    NLOptimizer::nloptUserEqConFunWrapper,
                    conFunc,
                    std::vector<double>(
                        tol.data(),
                        tol.data() + tol.rows() * tol.cols()));
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Adding user equality constraints"
                    << std::endl;
                return true;
            }
            catch (const std::exception &e)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Unable to bind constraints function\n"
                    << e.what()
                    << '\n';
                return false;
            }
        }

        /**
         * @brief Implementation of the optimization step
         *
         * @param x0 system's variables initial condition
         * @param u0 control action initial condition for warm start
         * @return Result<sizer.nu> optimization result
         */
        Result<sizer.nu> run(
            const cvec<sizer.nx> &x0,
            const cvec<sizer.nu> &u0)
        {
            checkOrQuit();

            Result<sizer.nu> r;

            std::vector<double> optX0;
            optX0.resize(((ph() * nx()) + (nu() * ch()) + 1));

            int counter = 0;
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    optX0[counter++] = x0(j);
                }
            }

            mat<sizer.ph, sizer.nu> Umv;
            Umv.resize(ph(), nu());
            Umv.setZero();

            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nu(); j++)
                {
                    Umv(i, j) = u0(j);
                }
            }

            cvec<(sizer.ph * sizer.nu)> UmvVectorized;
            UmvVectorized.resize((ph() * nu()));

            int vec_counter = 0;
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nu(); j++)
                {
                    UmvVectorized(vec_counter++) = Umv(i, j);
                }
            }

            cvec<(sizer.nu * sizer.ch)> res;
            res = mapping.Iu2z() * UmvVectorized;

            for (size_t j = 0; j < (nu() * ch()); j++)
            {
                optX0[counter++] = res(j);
            }

            optX0[((ph() * nx()) + (nu() * ch()) + 1) - 1] = currentSlack;

            try
            {
                auto res = innerOpt->optimize(optX0);

                cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> opt_vector;
                opt_vector = Eigen::Map<cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)>>(res.data(), res.size());

                r.cost = innerOpt->last_optimum_value();
                r.retcode = innerOpt->last_optimize_result();

                Logger::instance().log(Logger::log_type::INFO)
                    << "Optimization end after: "
                    << innerOpt->get_numevals()
                    << " evaluation steps"
                    << std::endl;
                Logger::instance().log(Logger::log_type::INFO)
                    << "Optimization end with code: "
                    << r.retcode
                    << std::endl;
                Logger::instance().log(Logger::log_type::INFO)
                    << "Optimization end with cost: "
                    << r.cost
                    << std::endl;

                mat<(sizer.ph + 1), sizer.nx> Xmat;
                Xmat.resize((ph() + 1), nx());

                mat<(sizer.ph + 1), sizer.nu> Umat;
                Umat.resize((ph() + 1), nu());

                mapping.unwrapVector(opt_vector, x0, Xmat, Umat, currentSlack);

                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Optimal predicted state vector\n"
                    << Xmat
                    << std::endl;
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Optimal predicted output vector\n"
                    << Umat
                    << std::endl;
                r.cmd = Umat.row(0);
            }
            catch (const std::exception &e)
            {
                Logger::instance().log(Logger::log_type::INFO)
                    << "No optimal solution found: "
                    << e.what()
                    << std::endl;
                r.cmd = last_r.cmd;
                r.retcode = -1;
            }

            last_r = r;
            return r;
        }

    private:
        /**
         * @brief Forward the objective function evaluation to the internal solver
         *
         * @param x current optimization vector
         * @param grad objective gradient w.r.t. the current optimization vector
         * @param objFunc reference to the objective class
         * @return double objective function value
         */
        static double nloptObjFunWrapper(
            const std::vector<double> &x,
            std::vector<double> &grad,
            void *objFunc)
        {
            bool hasGradient = !grad.empty();
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x_arr;
            x_arr = Eigen::Map<cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)>>((double *)x.data(), x.size());

            auto res = ((Objective<sizer> *)objFunc)->evaluate(x_arr, hasGradient);

            if (hasGradient)
            {
                // The gradient should be transposed since the difference between matlab and nlopt
                std::copy_n(
                    &res.grad.transpose()[0],
                    res.grad.cols() * res.grad.rows(),
                    grad.begin());
            }
            return res.value;
        }

        /**
         * @brief Forward the system's dynamics equality constraints evaluation to the internal solver
         *
         * @param result constraints value
         * @param n dimension of the optimization vector
         * @param x current optimization vector
         * @param grad equality constraints gradient w.r.t. the current optimization vector
         * @param conFunc reference to the constraints class
         */
        static void nloptEqConFunWrapper(
            unsigned int,
            double *result,
            unsigned int n,
            const double *x,
            double *grad,
            void *conFunc)
        {
            bool hasGradient = (grad != NULL);
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x_arr;
            x_arr = Eigen::Map<cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)>>((double *)x, n);

            auto res = static_cast<Constraints<sizer> *>(conFunc)->evaluateStateModelEq(x_arr, hasGradient);

            std::memcpy(
                result,
                res.value.data(),
                res.value.size() * sizeof(double));

            if (hasGradient)
            {
                // The gradient should be transposed since the difference between matlab and nlopt
                std::memcpy(
                    grad,
                    res.grad.transpose().data(),
                    res.grad.rows() * res.grad.cols() * sizeof(double));
            }
        }

        /**
         * @brief Forward the user inequality constraints evaluation to the internal solver
         *
         * @param result constraints value
         * @param n dimension of the optimization vector
         * @param x current optimization vector
         * @param grad equality constraints gradient w.r.t. the current optimization vector
         * @param conFunc reference to the constraints class
         */
        static void nloptUserIneqConFunWrapper(
            unsigned int,
            double *result,
            unsigned int n,
            const double *x,
            double *grad,
            void *conFunc)
        {
            bool hasGradient = (grad != NULL);
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x_arr;
            x_arr = Eigen::Map<cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)>>((double *)x, n);

            auto res = static_cast<Constraints<sizer> *>(conFunc)->evaluateIneq(x_arr, hasGradient);

            std::memcpy(
                result,
                res.value.data(),
                res.value.size() * sizeof(double));

            if (hasGradient)
            {
                // The gradient should be transposed since the difference between matlab and nlopt
                std::memcpy(
                    grad,
                    res.grad.transpose().data(),
                    res.grad.rows() * res.grad.cols() * sizeof(double));
            }
        }

        /**
         * @brief Forward the user equality constraints evaluation to the internal solver
         *
         * @param result constraints value
         * @param n dimension of the optimization vector
         * @param x current optimization vector
         * @param grad equality constraints gradient w.r.t. the current optimization vector
         * @param conFunc reference to the constraints class
         */
        static void nloptUserEqConFunWrapper(
            unsigned int /*m*/,
            double *result,
            unsigned int n,
            const double *x,
            double *grad,
            void *conFunc)
        {
            bool hasGradient = (grad != NULL);
            cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> x_arr;
            x_arr = Eigen::Map<cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)>>((double *)x, n);

            auto res = static_cast<Constraints<sizer> *>(conFunc)->evaluateEq(x_arr, hasGradient);

            std::memcpy(
                result,
                res.value.data(),
                res.value.size() * sizeof(double));

            if (hasGradient)
            {
                // The gradient should be transposed since the difference between matlab and nlopt
                std::memcpy(
                    grad,
                    res.grad.transpose().data(),
                    res.grad.rows() * res.grad.cols() * sizeof(double));
            }
        }

        nlopt::opt *innerOpt;
        Result<sizer.nu> last_r;
        double currentSlack;
        bool hard;

        Mapping<sizer> mapping;
    };
} // namespace mpc
