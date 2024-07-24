/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/NLMPC/Constraints.hpp>
#include <mpc/IOptimizer.hpp>
#include <mpc/Logger.hpp>
#include <mpc/NLMPC/Mapping.hpp>
#include <mpc/NLMPC/Objective.hpp>
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

        using IOptimizer<sizer>::currentSlack;
        using IOptimizer<sizer>::hard;
        using IOptimizer<sizer>::result;
        using IOptimizer<sizer>::sequence;

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
        void onInit() override
        {
            innerOpt = new nlopt::opt(nlopt::LD_SLSQP, ((ph() * nx()) + (nu() * ch()) + 1));

            COND_RESIZE_CVEC(sizer,lb,((ph() * nx()) + (nu() * ch()) + 1));
            lb.setConstant(-std::numeric_limits<float>::infinity());

            COND_RESIZE_CVEC(sizer,ub, ((ph() * nx()) + (nu() * ch()) + 1));
            ub.setConstant(std::numeric_limits<float>::infinity());

            COND_RESIZE_CVEC(sizer,opt_vector, ((ph() * nx()) + (nu() * ch()) + 1));
            opt_vector.setZero();

            setParameters(NLParameters());

            COND_RESIZE_CVEC(sizer,result.cmd, nu());
            result.cmd.setZero();

            COND_RESIZE_MAT(sizer,sequence.state,ph() + 1, nx());
            sequence.state.setZero();
            COND_RESIZE_MAT(sizer,sequence.input,ph() + 1, nu());
            sequence.input.setZero();
            COND_RESIZE_MAT(sizer,sequence.output,ph() + 1, ny());
            sequence.output.setZero();

            currentSlack = 0;
            is_first_iteration = true;
        }

        /**
         * @brief Set the model and the mapping object references
         *
         * @param sysModel the model object
         * @param map the mapping object
         */
        void setModel(std::shared_ptr<Model<sizer>> sysModel, std::shared_ptr<Mapping<sizer>> map)
        {
            checkOrQuit();

            mapping = map;
            model = sysModel;
        }

        /**
         * @brief Set the Cost And Constraints object
         *
         * @param objFunc
         * @param conFunc
         */
        void setCostAndConstraints(
            std::shared_ptr<Objective<sizer>> objFunc,
            std::shared_ptr<Constraints<sizer>> conFunc)
        {
            checkOrQuit();

            this->objFunc = objFunc;
            this->conFunc = conFunc;
        }

        /**
         * @brief Set the optmiziation parameters
         *
         * @param param parameters desired
         */
        void setParameters(const Parameters &param) override
        {
            checkOrQuit();

            auto nl_param = dynamic_cast<const NLParameters *>(&param);

            innerOpt->set_ftol_rel(nl_param->relative_ftol);
            innerOpt->set_xtol_rel(nl_param->relative_xtol);
            innerOpt->set_ftol_abs(nl_param->absolute_ftol);
            innerOpt->set_xtol_abs(nl_param->absolute_xtol);

            innerOpt->set_x_weights(1.0);

            if (nl_param->time_limit > 0)
            {
                innerOpt->set_maxtime(nl_param->time_limit);
            }

            innerOpt->set_maxeval(nl_param->maximum_iteration);

            // print the parameters
            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting relative function tolerance: "
                << nl_param->relative_ftol << ", internal value: "
                << innerOpt->get_ftol_rel()
                << std::endl;

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting relative variable tolerance: "
                << nl_param->relative_xtol << ", internal value: "
                << innerOpt->get_xtol_rel()
                << std::endl;

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting absolute function tolerance: "
                << nl_param->absolute_ftol << ", internal value: "
                << innerOpt->get_ftol_abs()
                << std::endl;

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting maximum number of function evaluations: "
                << nl_param->maximum_iteration << ", internal value: "
                << innerOpt->get_maxeval()
                << std::endl;

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting maximum time limit: "
                << nl_param->time_limit << ", internal value: "
                << innerOpt->get_maxtime()
                << std::endl;

            // set the bounds for the slack variable
            // in case of hard constraints we are forcing the slack variable to be 0
            if(nl_param->hard_constraints)
            {
                lb[((ph() * nx()) + (nu() * ch()) + 1) - 1] = 0;
                ub[((ph() * nx()) + (nu() * ch()) + 1) - 1] = 0;
            }

            enable_warm_start = nl_param->enable_warm_start;

            updateBounds();

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting tolerances and stopping criterias"
                << std::endl;
        }

        /**
         * @brief Bind the objective function class with the internal solver
         * objective function referemce
         *
         * @return true
         * @return false
         */
        bool bindObjective()
        {
            checkOrQuit();

            try
            {
                innerOpt->set_min_objective(NLOptimizer::nloptObjFunWrapper, objFunc.get());
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
         * system's dynamics equality constraints function reference
         *
         * @param tol equality constraints tolerances
         * @return true
         * @return false
         */
        bool bindEq(
            constraints_type,
            const cvec<(sizer.ph * sizer.nx)> tol)
        {
            checkOrQuit();

            try
            {
                innerOpt->add_equality_mconstraint(
                    NLOptimizer::nloptEqConFunWrapper,
                    conFunc.get(),
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
         * @param tol inequality constraints tolerances
         * @return true
         * @return false
         */
        bool bindUserIneq(
            constraints_type,
            const cvec<sizer.ineq> tol)
        {
            checkOrQuit();

            try
            {
                innerOpt->add_inequality_mconstraint(
                    NLOptimizer::nloptUserIneqConFunWrapper,
                    conFunc.get(),
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
         * @param type constraints type
         * @param tol equality constraints tolerances
         * @return true
         * @return false
         */
        bool bindUserEq(
            constraints_type /*type*/,
            const cvec<sizer.eq> tol)
        {
            checkOrQuit();

            try
            {
                innerOpt->add_equality_mconstraint(
                    NLOptimizer::nloptUserEqConFunWrapper,
                    conFunc.get(),
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
         * @brief Sets the state bounds for the NLOptimizer.
         *
         * This function sets the lower and upper bounds for the state variables of the NLOptimizer.
         * The bounds are used to constrain the state variables during the optimization process.
         *
         * @param lb The lower bounds for the state variables.
         * @param ub The upper bounds for the state variables.
         * @param slice The slice of the bounds to set.
         * @return True if the state bounds were successfully set, false otherwise.
         */
        bool setStateBounds(
            const cvec<sizer.nx> &lower_bounds,
            const cvec<sizer.nx> &upper_bounds,
            const HorizonSlice& slice)
        {
            checkOrQuit();

            // if the slice is (-1, -1) set the bounds for the entire state vector
            size_t start = slice.start == -1 ? 0 : slice.start;
            size_t end = slice.end == -1 ? ph() : slice.end;

            // replicate the state bounds for the prediction horizon
            for (size_t i = start; i < end; i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    lb[(i * nx()) + j] = lower_bounds(j);
                    ub[(i * nx()) + j] = upper_bounds(j);
                }
            }

            updateBounds();

            return true;
        }

        /**
         * Sets the input bounds for the NLOptimizer.
         *
         * @param lb The lower bounds for the input variables.
         * @param ub The upper bounds for the input variables.
         * @param slice The slice of the input bounds to set.
         * @return True if the input bounds were successfully set, false otherwise.
         */
        bool setInputBounds(
            const cvec<sizer.nu> &lower_bounds,
            const cvec<sizer.nu> &upper_bounds,
            const HorizonSlice& slice)
        {
            checkOrQuit();

            // if the slice is (-1, -1) set the bounds for the entire input vector
            size_t start = slice.start == -1 ? 0 : slice.start;
            size_t end = slice.end == -1 ? ch() : slice.end;

            // replicate the input bounds for the control horizon
            for (size_t i = start; i < end; i++)
            {
                for (size_t j = 0; j < nu(); j++)
                {
                    lb[(ph() * nx()) + (i * nu()) + j] = lower_bounds(j);
                    ub[(ph() * nx()) + (i * nu()) + j] = upper_bounds(j);
                }
            }

            updateBounds();

            return true;
        }

        /**
         * @brief Implementation of the optimization step
         *
         * @param x0 system's variables initial condition
         * @param u0 control action initial condition for warm start
         */
        void run(
            const cvec<sizer.nx> &x0,
            const cvec<sizer.nu> &u0) override
        {
            checkOrQuit();

            Result<sizer.nu> r;

            std::vector<double> optX0;
            optX0.assign(((ph() * nx()) + (nu() * ch()) + 1),0.0);

            if(is_first_iteration || !enable_warm_start)
            {
                // the whole optimization vector is initialized with the initial state x0
                // and the initial control action u0 for the full prediction horizon
                // this has to be done only for the first iteration or if the warm start is disabled
                for (size_t i = 0; i < ph(); i++)
                {
                    for (size_t j = 0; j < nx(); j++)
                    {
                        opt_vector[(i * nx()) + j] = x0(j);
                    }
                }

                for (size_t i = 0; i < ch(); i++)
                {
                    for (size_t j = 0; j < nu(); j++)
                    {
                        opt_vector[(ph() * nx()) + (i * nu()) + j] = u0(j);
                    }
                }
            }
            
            // fill the remaining elements with the previous state starting from
            // the third element of the prediction horizon 
            // (we shift the sequence to the left by one step)
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nx(); j++)
                {
                    if(i == ph() - 1)
                    {
                        optX0[(i * nx()) + j] = opt_vector[(i * nx()) + j];
                    }
                    else{
                        optX0[(i * nx()) + j] = opt_vector[((i+1) * nx()) + j];
                    }
                }
            }

            cvec<(sizer.ph * sizer.nu)> UmvVectorized;
            COND_RESIZE_CVEC(sizer,UmvVectorized,(ph() * nu()));

            // convert from the optimized vector to the vectorized manipulated variable
            // using the mapping matrix Iz2u
            cvec<(sizer.ph * sizer.nu)> tmp_mult;
            tmp_mult = mapping->Iz2u() * opt_vector.middleRows((ph() * nx()), (nu() * ch()));

            // fill the remaining elements with the previous control action starting from
            // the third element of the control horizon
            // (we shift the sequence to the left by one step)
            for (size_t i = 0; i < ph(); i++)
            {
                for (size_t j = 0; j < nu(); j++)
                {
                    if (i == ph() - 1)
                    {
                        UmvVectorized[(i * nu()) + j] = tmp_mult[(i * nu()) + j];
                    }
                    else
                    {
                        UmvVectorized[(i * nu()) + j] = tmp_mult[((i+1) * nu()) + j];
                    }
                }
            }

            cvec<(sizer.nu * sizer.ch)> res;
            res = mapping->Iu2z() * UmvVectorized;

            // put the control action back in the optimization vector
            for (size_t i = 0; i < (nu() * ch()); i++)
            {
                optX0[(ph() * nx()) + i] = res(i);
            }

            // put the slack variable in the optimization vector
            optX0[((ph() * nx()) + (nu() * ch()) + 1) - 1] = currentSlack;

            // let's start the optimization
            bool optimizationSuccess = false;

            try
            {
                std::vector<double> opt_v = innerOpt->optimize(optX0);
                // convert from std vector to eigen vector by copying the data
                Eigen::Map<cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)>>(opt_v.data(), opt_v.size()).swap(opt_vector);
                optimizationSuccess = true;
                is_first_iteration = false;

                // check if the solution vector is feasible or not
                r.is_feasible = conFunc->isFeasible(opt_vector);

                if (r.is_feasible)
                {
                    Logger::instance().log(Logger::log_type::DETAIL)
                        << "Optimal solution found"
                        << std::endl;
                }
                else
                {
                    Logger::instance().log(Logger::log_type::DETAIL)
                        << "Optimal solution found but not feasible"
                        << std::endl;
                }
            }
            catch (nlopt::roundoff_limited &e)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "No optimal solution found: "
                    << e.what()
                    << std::endl;

                // we reached floating point precision limit before reaching the stopping criteria
                optimizationSuccess = false;
                r.solver_status_msg = "Floating point precision limit reached before stopping criteria";
            }
            catch (const std::exception &e)
            {
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "No optimal solution found: "
                    << e.what()
                    << std::endl;

                // generic exception handling
                optimizationSuccess = false;
                r.solver_status_msg = "Internal solver error: " + std::string(e.what());
            }

            if (optimizationSuccess)
            {
                r.cost = innerOpt->last_optimum_value();
                r.solver_status = innerOpt->last_optimize_result();
                // convert from nlopt result code to ResultStatus enum
                r.status = convertToResultStatus(r.solver_status);

                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Optimization end after: "
                    << innerOpt->get_numevals()
                    << " evaluation steps"
                    << std::endl;

                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Optimization end with code: "
                    << r.solver_status
                    << std::endl;

                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Optimization end with cost: "
                    << r.cost
                    << std::endl;

                mat<(sizer.ph + 1), sizer.nx> Xmat;
                COND_RESIZE_MAT(sizer,Xmat,(ph() + 1), nx());

                mat<(sizer.ph + 1), sizer.nu> Umat;
                COND_RESIZE_MAT(sizer,Umat,(ph() + 1), nu());

                mapping->unwrapVector(opt_vector, x0, Xmat, Umat, currentSlack);

                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Optimal predicted state vector\n"
                    << Xmat
                    << std::endl;
                Logger::instance().log(Logger::log_type::DETAIL)
                    << "Optimal predicted control input vector\n"
                    << Umat
                    << std::endl;

                r.cmd = Umat.row(0);

                sequence.state = Xmat.block(0, 0, ph()+1, nx());
                sequence.input = Umat.block(0, 0, ph()+1, nu());
                sequence.output = model->getOutput(Xmat, Umat).block(0, 0, ph()+1, ny());
            }
            else
            {
                r.cost = mpc::inf;
                r.cmd = result.cmd;
                r.solver_status = -1;
                // set the result status to error
                r.status = ResultStatus::ERROR;

                sequence.state.setZero();
                sequence.input.setZero();
                sequence.output.setZero();
            }

            // update the result
            result = r;
        }

        /**
         * @brief Get the lower bound of the optimization variables
         *
         * @return lower bound
         */
        mpc::cvec<(sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1> getLowerBound() const
        {
            return lb;
        }

        /**
         * @brief Get the upper bound of the optimization variables
         *
         * @return upper bound
         */
        mpc::cvec<(sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1> getUpperBound() const
        {
            return ub;
        }

    private:
        /**
         * @brief Update the bounds for the internal solver
         */
        void updateBounds()
        {
            // convert from eigen vector to std vector
            std::vector<double> lb_vec(lb.data(), lb.data() + lb.rows() * lb.cols());
            std::vector<double> ub_vec(ub.data(), ub.data() + ub.rows() * ub.cols());

            innerOpt->set_lower_bounds(lb_vec);
            innerOpt->set_upper_bounds(ub_vec);

            // get the bounds from the solver
            auto lb_solver = innerOpt->get_lower_bounds();
            auto ub_solver = innerOpt->get_upper_bounds();

            // print the bounds
            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting lower bounds: "
                << std::endl;
            std::stringstream ss_lb;
            ss_lb << "\n";
            for (size_t i = 0; i < lb_solver.size(); i++)
            {
                ss_lb << lb_solver[i] << "\n";
            }
            Logger::instance().log(Logger::log_type::DETAIL) << ss_lb.str() << "\n";

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting upper bounds: "
                << std::endl;
            std::stringstream ss_ub;
            ss_ub << "\n";
            for (size_t i = 0; i < ub_solver.size(); i++)
            {
                ss_ub << ub_solver[i] << "\n";
            }
            Logger::instance().log(Logger::log_type::DETAIL) << ss_ub.str() << "\n";

        }

        /**
         * @brief Converts an integer value to the corresponding ResultStatus enum value.
         *
         * This function maps the given integer value to the corresponding ResultStatus enum value.
         * If the integer value does not match any known result, the function returns ResultStatus::UNKNOWN.
         *
         * @param status The integer value to convert.
         * @return The corresponding ResultStatus enum value.
         *
         * @see ResultStatus
         */
        ResultStatus convertToResultStatus(int status)
        {
            switch (status)
            {
            case nlopt::FAILURE:
            case nlopt::INVALID_ARGS:
            case nlopt::OUT_OF_MEMORY:
            case nlopt::ROUNDOFF_LIMITED:
            case nlopt::FORCED_STOP:
                return ResultStatus::ERROR;
            case nlopt::SUCCESS:
            case nlopt::STOPVAL_REACHED:
            case nlopt::FTOL_REACHED:
            case nlopt::XTOL_REACHED:
                return ResultStatus::SUCCESS;
            case nlopt::MAXEVAL_REACHED:
            case nlopt::MAXTIME_REACHED:
                return ResultStatus::MAX_ITERATION;
            default:
                return ResultStatus::UNKNOWN;
            }
        }

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

        std::shared_ptr<Objective<sizer>> objFunc;
        std::shared_ptr<Constraints<sizer>> conFunc;
        std::shared_ptr<Mapping<sizer>> mapping;
        std::shared_ptr<Model<sizer>> model;

        cvec<(sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1>  lb, ub;
        cvec<((sizer.ph * sizer.nx) + (sizer.nu * sizer.ch) + 1)> opt_vector;
        bool is_first_iteration = true;
        bool enable_warm_start = false;
    };
} // namespace mpc
