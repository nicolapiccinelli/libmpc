/*
 *   Copyright (c) 2023-2026 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IComponent.hpp>

#include <mpc/NLMPC/Model.hpp>
#include <mpc/NLMPC/Mapping.hpp>

namespace mpc
{
    /**
     * @brief Abstract class defining the optimizer interface for
     * linear and non-linear MPC
     *
     * @tparam Tnx dimension of the state space
     * @tparam Tnu dimension of the input space
     * @tparam Tndu dimension of the measured disturbance space
     * @tparam Tny dimension of the output space
     * @tparam Tph length of the prediction horizon
     * @tparam Tch length of the control horizon
     * @tparam Tineq number of the user inequality constraints
     * @tparam Teq number of the user equality constraints
     */
    template <MPCSize sizer>
    class NLOptimizerBase : public IOptimizer<sizer>
    {
    public:
        virtual ~NLOptimizerBase() {}

        /**
         * @brief Initialization hook override used to perform the optimizer interfaces
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        virtual void onInit() override = 0;

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
        void setCostAndConstraints(std::shared_ptr<Objective<sizer>> objFunc, std::shared_ptr<Constraints<sizer>> conFunc)
        {
            checkOrQuit();

            this->objFunc = objFunc;
            this->conFunc = conFunc;
        }

        /**
         * @brief Bind the objective function class with the internal solver
         * objective function referemce
         *
         * @return true
         * @return false
         */
        virtual bool bindObjective() = 0;

        /**
         * @brief Bind the constraints class with the internal solver
         * system's dynamics equality constraints function reference
         *
         * @param tol equality constraints tolerances
         * @return true
         * @return false
         */
        virtual bool bindEq(ConstraintsType, const cvec<(sizer.ph * sizer.nx)> tol) = 0;

        /**
         * @brief Bind the constraints class with the internal solver
         * user inequality constraints function referemce
         *
         * @param tol inequality constraints tolerances
         * @return true
         * @return false
         */
        virtual bool bindUserIneq(ConstraintsType, const cvec<sizer.ineq> tol) = 0;

        /**
         * @brief Bind the constraints class with the internal solver
         * user equality constraints function referemce
         *
         * @param type constraints type
         * @param tol equality constraints tolerances
         * @return true
         * @return false
         */
        virtual bool bindUserEq(ConstraintsType, const cvec<sizer.eq> tol) = 0;        

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
        virtual bool setStateBounds(const cvec<sizer.nx> &lower_bounds, const cvec<sizer.nx> &upper_bounds, const HorizonSlice &slice) = 0;

        /**
         * Sets the input bounds for the NLOptimizer.
         *
         * @param lb The lower bounds for the input variables.
         * @param ub The upper bounds for the input variables.
         * @param slice The slice of the input bounds to set.
         * @return True if the input bounds were successfully set, false otherwise.
         */
        virtual bool setInputBounds(const cvec<sizer.nu> &lower_bounds, const cvec<sizer.nu> &upper_bounds, const HorizonSlice& slice) = 0;

        /**
         * @brief Abstract setter for the optimizer parameters
         *
         * @param param desired parameters (refers to the optimizer specific for the meaning of the parameters)
         */
        virtual void setParameters(const Parameters &param) override = 0;

        /**
         * @brief Abstract caller to perform the optimization
         *
         * @param x0 system's variables initial condition
         * @param u0 control action initial condition for warm start
         * @return Result<Tnu> optimization result
         */
        virtual void run(const cvec<sizer.nx> &x0, const cvec<sizer.nu> &u0) override = 0;

    protected:
        double currentSlack;
        bool hard;

        std::shared_ptr<Mapping<sizer>> mapping;
        std::shared_ptr<Model<sizer>> model;
        std::shared_ptr<Objective<sizer>> objFunc;
        std::shared_ptr<Constraints<sizer>> conFunc;

        using IComponent<sizer>::checkOrQuit;
    };
}