/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IMPC.hpp>

#include <mpc/NLMPC/Constraints.hpp>
#include <mpc/NLMPC/Objective.hpp>
#include <mpc/NLMPC/NLOptimizer.hpp>

namespace mpc
{
    /**
     * @brief Non-lnear MPC front-end class
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
        int Tnx = Eigen::Dynamic, int Tnu = Eigen::Dynamic, int Tny = Eigen::Dynamic,
        int Tph = Eigen::Dynamic, int Tch = Eigen::Dynamic,
        int Tineq = Eigen::Dynamic, int Teq = Eigen::Dynamic>
    class NLMPC : public IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>
    {

    private:
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::optPtr;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::setDimension;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::nu;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::nx;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::ndu;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::ny;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::ph;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::ch;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::ineq;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::eq;

    public:
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::optimize;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::setLoggerLevel;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::setLoggerPrefix;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::getLastResult;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::isSliceUnset;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::isPredictionHorizonSliceValid;
        using IMPC<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::isControlHorizonSliceValid;

        NLMPC()
        {
            buildInternalModules();
            setDimension();
        }

        NLMPC(
            const int &nx, const int &nu, const int &ny,
            const int &ph, const int &ch, const int &ineq, const int &eq)
        {
            buildInternalModules();
            setDimension(nx, nu, 0, ny, ph, ch, ineq, eq);
        }

        ~NLMPC()
        {
            delete optPtr;
        }

        /**
         * @brief Set the discretization time step to use for numerical integration
         *
         * @param ts sample time in seconds
         * @return true
         * @return false
         */
        bool setDiscretizationSamplingTime(const double ts) override
        {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting sampling time to: "
                << ts
                << " sec(s)"
                << std::endl;

            auto res = model->setContinuous(true, ts);
            return res;
        }

        /**
         * @brief  Set the solver specific parameters
         *
         * @param param desired parameters (the structure must be of type NLParameters)
         */
        void setOptimizerParameters(const Parameters &param) override
        {
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setParameters(param);
        }

        /**
         *
         * @brief Set the scaling factor for the control input
         *
         * @param scaling scaling vector
         */
        void setInputScale(const cvec<Tnu> scaling) override
        {
            mapping->setInputScaling(scaling);

            objF->setModel(model, mapping);
            conF->setModel(model, mapping);
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setModel(model, mapping);
        }

        /**
         *
         * @brief Set the scaling factor for the dynamical system's states variables
         *
         * @param scaling scaling vector
         */
        void setStateScale(const cvec<Tnx> scaling) override
        {
            mapping->setStateScaling(scaling);

            objF->setModel(model, mapping);
            conF->setModel(model, mapping);
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setModel(model, mapping);
        }

        /**
         * @brief Set the handler to the function defining the objective function
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setObjectiveFunction(const typename IDimensionable<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::ObjFunHandle handle)
        {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting objective function handle"
                << std::endl;

            auto res = objF->setObjective(handle);

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Binding objective function handle"
                << std::endl;

            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->bindObjective();
            return res;
        }

        /**
         * @brief Set the handler to the function defining the state space update function.
         * Based on the type of system (continuous or discrete) you should provide the appropriate
         * vector field differential equations or the finite differences update model
         *
         * @param handle function handler
         * @param eq_tol equality constraints tolerances (default 1e-10)
         * @return true
         * @return false
         */
        bool setStateSpaceFunction(const typename IDimensionable<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::StateFunHandle handle,
                                   const float eq_tol = 1e-10)
        {
            cvec<2 * Size(Tph) * Size(Tny)> ineq_tol_vec;
            COND_RESIZE_CVEC(MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq), ineq_tol_vec, (2 * ph() * ny()));
            ineq_tol_vec.setOnes();

            cvec<(Size(Tph) * Size(Tnx))> eq_tol_vec;
            COND_RESIZE_CVEC(MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq), eq_tol_vec, (ph() * nx()));
            eq_tol_vec.setOnes();

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting state space function handle"
                << std::endl;

            bool res = model->setStateModel(handle);

            objF->setModel(model, mapping);
            conF->setModel(model, mapping);
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setModel(model, mapping);

            Logger::instance().log(Logger::log_type::DETAIL)
                << "Binding state space constraints"
                << std::endl;

            res = res & ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->bindEq(constraints_type::EQ, eq_tol_vec * eq_tol);

            return res;
        }

        /**
         * Set the handler to the function defining the output function
         *
         * @param handle function handler
         * @return true
         * @return false
         */
        bool setOutputFunction(const typename IDimensionable<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::OutFunHandle handle)
        {
            Logger::instance().log(Logger::log_type::DETAIL)
                << "Setting output function handle"
                << std::endl;

            bool res = model->setOutputModel(handle);

            objF->setModel(model, mapping);
            conF->setModel(model, mapping);
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setModel(model, mapping);

            return res;
        }

        /**
         * @brief Set the handler to the function defining the user inequality constraints
         * These constraints are custom, and the user must provide the function handler
         * that will be used to evaluate the constraints. These constraints could be not
         * satisfied by the optimization algorithm during the optimization process.
         *
         * @param handle function handler
         * @param tol inequality constraints tolerances (default 1e-10)
         * @return true
         * @return false
         */
        bool setIneqConFunction(
            const typename IDimensionable<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::IConFunHandle handle, const float tol = 1e-10)
        {
            cvec<Tineq> tol_vec;
            tol_vec = cvec<Tineq>::Ones(ineq());

            auto res = conF->setIneqConstraints(handle, tol);
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->bindUserIneq(constraints_type::UINEQ, tol_vec * tol);
            return res;
        }

        /**
         * @brief Set the handler to the function defining the user equality constraints
         * These constraints are custom, and the user must provide the function handler
         * that will be used to evaluate the constraints. These constraints could not be satisfied
         * by the optimization algorithm during the optimization process.
         *
         * @param handle function handler
         * @param tol equality constraints tolerances (default 1e-10)
         * @return true
         * @return false
         */
        bool setEqConFunction(
            const typename IDimensionable<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>::EConFunHandle handle, const float tol = 1e-10)
        {
            cvec<Teq> tol_vec;
            tol_vec = cvec<Teq>::Ones(Size(Teq));

            auto res = conF->setEqConstraints(handle, tol);
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->bindUserEq(constraints_type::UEQ, tol_vec * tol);
            return res;
        }

        /**
         * @brief Set the state constraints, on the entire horizon length.
         * These constraints are defined as box constraints for the state, input, and output variables
         * and are used to restrict the search space of the optimization problem. Thus, they are
         * always satisfied by the optimizer during the optimization process.
         *
         * @param XMinMat the minimum state constraints matrix
         * @param XMaxMat the maximum state constraints matrix
         */
        bool setStateBounds(const mat<Tnx, Tph> &XMinMat, const mat<Tnx, Tph> &XMaxMat) override
        {
            // set the state bounds to the optimizer
            bool res = true;

            // iterate over the prediction horizon in the matrices and set the constraints
            for (size_t i = 0; i < ph(); i++)
            {
                int index = (int)i;
                HorizonSlice slice = {index, index + 1};
                res &= ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setStateBounds(XMinMat.col(i), XMaxMat.col(i), slice);
            }

            return res;
        }

        /**
         * @brief Set the input constraints on the entire horizon length.
         * These constraints are defined as box constraints for the state, input, and output variables
         * and are used to restrict the search space of the optimization problem. Thus, they are
         * always satisfied by the optimizer during the optimization process.
         *
         * @param UMinMat the minimum input constraints matrix
         * @param UMaxMat the maximum input constraints matrix
         */
        bool setInputBounds(const mat<Tnu, Tch> &UMinMat, const mat<Tnu, Tch> &UMaxMat) override
        {
            // set the input bounds to the optimizer
            bool res = true;

            // iterate over the control horizon in the matrices and set the constraints
            for (size_t i = 0; i < ch(); i++)
            {
                int index = (int) i;
                HorizonSlice slice = {index, index + 1};
                res &= ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setInputBounds(UMinMat.col(i), UMaxMat.col(i), slice);
            }

            return res;
        }

        /**
         * @brief Set the output constraints on the entire horizon length.
         * These constraints are defined as box constraints for the state, input, and output variables
         * and are used to restrict the search space of the optimization problem. Thus, they are
         * always satisfied by the optimizer during the optimization process.
         *
         * @param YMinMat the minimum output constraints matrix
         * @param YMaxMat the maximum output constraints matrix
         */
        bool setOutputBounds(const mat<Tny, Tph> &/*YMinMat*/, const mat<Tny, Tph> &/*YMaxMat*/) override
        {
            Logger::instance().log(Logger::log_type::ERROR)
                << "Output constraints cannot be set for this type of MPC, the ouput is not\
                considered in the optimization process. Thus we cannot restrict the search space."
                << std::endl;

            throw std::runtime_error("Output constraints cannot be set for this type of MPC");
        }

        /**
         * @brief Set the state constraints on a certain slice of the horizon.
         * These constraints are defined as box constraints for the state, input, and output variables
         * and are used to restrict the search space of the optimization problem. Thus, they are
         * always satisfied by the optimizer during the optimization process.
         *
         * @param XMin the minimum state constraints vector
         * @param XMax the maximum state constraints vector
         * @param slice the slice of the horizon to apply the constraints to
         */
        bool setStateBounds(const cvec<Tnx> &XMin, const cvec<Tnx> &XMax, const HorizonSlice &slice) override
        {
            // set the state bounds to the optimizer
            bool res = true;

            res = isSliceUnset(slice) || isPredictionHorizonSliceValid(slice);
            if (res)
            {
                res &= ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setStateBounds(XMin, XMax, slice);
            }

            return res;
        }

        /**
         * @brief Set the input constraints on a certain slice of the horizon.
         * These constraints are defined as box constraints for the state, input, and output variables
         * and are used to restrict the search space of the optimization problem. Thus, they are
         * always satisfied by the optimizer during the optimization process.
         *
         * @param UMin the minimum input constraints vector
         * @param UMax the maximum input constraints vector
         * @param slice the slice of the horizon to apply the constraints to
         */
        bool setInputBounds(const cvec<Tnu> &UMin, const cvec<Tnu> &UMax, const HorizonSlice &slice) override
        {
            // set the input bounds to the optimizer
            bool res = true;

            res = isSliceUnset(slice) || isControlHorizonSliceValid(slice);
            if (res)
            {
                res &= ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setInputBounds(UMin, UMax, slice);
            }

            return res;
        }

        /**
         * @brief Set the output constraints on a certain slice of the horizon.
         * These constraints are defined as box constraints for the state, input, and output variables
         * and are used to restrict the search space of the optimization problem. Thus, they are
         * always satisfied by the optimizer during the optimization process.
         *
         * @param YMin the minimum output constraints vector
         * @param YMax the maximum output constraints vector
         * @param slice the slice of the horizon to apply the constraints to
         */
        bool setOutputBounds(const cvec<Tny> &/*YMin*/, const cvec<Tny> &/*YMax*/, const HorizonSlice &/*slice*/) override
        {
            Logger::instance().log(Logger::log_type::ERROR)
                << "Output constraints cannot be set for this type of MPC, the ouput is not\
                considered in the optimization process. Thus we cannot restrict the search space."
                << std::endl;

            throw std::runtime_error("Output constraints cannot be set for this type of MPC");
        }

    protected:
        /**
         * @brief Initilization hook for the interface
         */
        void onSetup() override
        {
            conF->initialize(
                nx(), nu(), 0, ny(),
                ph(), ch(), ineq(),
                eq());

            model->initialize(
                nx(), nu(), 0, ny(),
                ph(), ch(), ineq(),
                eq());

            mapping->initialize(
                nx(), nu(), 0, ny(),
                ph(), ch(), ineq(),
                eq());

            objF->initialize(
                nx(), nu(), 0, ny(),
                ph(), ch(), ineq(),
                eq());

            optPtr = new NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>();
            optPtr->initialize(
                nx(), nu(), 0, ny(),
                ph(), ch(), ineq(),
                eq());

            // set the model and mapping to the internal modules to allow the computation of the
            // objective function and constraints functions
            objF->setModel(model, mapping);
            conF->setModel(model, mapping);
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setModel(model, mapping);

            // set the objective function and the constraints functions to the optimizer
            ((NLOptimizer<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)> *)optPtr)->setCostAndConstraints(objF, conF);

            Logger::instance().log(Logger::log_type::INFO)
                << "Mapping assignment done"
                << std::endl;
        }

        /**
         * @brief Dynamical system initial condition update hook
         */
        void onModelUpdate(const cvec<Tnx> x0) override
        {
            objF->setCurrentState(x0);
            conF->setCurrentState(x0);
        }

    private:
        void buildInternalModules()
        {
            objF = std::make_shared<Objective<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>>();
            conF = std::make_shared<Constraints<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>>();
            model = std::make_shared<Model<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>>();
            mapping = std::make_shared<Mapping<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>>();
        }

        std::shared_ptr<Objective<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>> objF;
        std::shared_ptr<Constraints<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>> conF;
        std::shared_ptr<Model<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>> model;
        std::shared_ptr<Mapping<MPCSize(Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq)>> mapping;
    };

} // namespace mpc
