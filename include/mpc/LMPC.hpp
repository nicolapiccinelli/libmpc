/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IMPC.hpp>
#include <mpc/LMPC/LOptimizer.hpp>
#include <mpc/LMPC/ProblemBuilder.hpp>

namespace mpc
{
    /**
     * @brief Linear MPC front-end class
     *
     * @tparam Tnx dimension of the state space
     * @tparam Tnu dimension of the input space
     * @tparam Tndu dimension of the measured disturbance space
     * @tparam Tny dimension of the output space
     * @tparam Tph length of the prediction horizon
     * @tparam Tch length of the control horizon
     */
    template <
        int Tnx = Eigen::Dynamic, int Tnu = Eigen::Dynamic, int Tndu = Eigen::Dynamic,
        int Tny = Eigen::Dynamic, int Tph = Eigen::Dynamic, int Tch = Eigen::Dynamic>
    class LMPC : public IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>
    {

    private:
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::optPtr;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::setDimension;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::nu;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::nx;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::ndu;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::ny;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::ph;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::ch;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::ineq;
        using IDimensionable<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::eq;

    public:
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::optimize;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::setLoggerLevel;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::setLoggerPrefix;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::getLastResult;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::isSliceUnset;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::isPredictionHorizonSliceValid;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::isControlHorizonSliceValid;

    public:
        LMPC()
        {
            setDimension();
        }

        LMPC(
            const int &nx, const int &nu, const int &ndu,
            const int &ny, const int &ph, const int &ch)
        {
            setDimension(nx, nu, ndu, ny, ph, ch);
        }

        ~LMPC() = default;

        /**
         * @brief (NOT AVAILABLE) Set the discretization time step to use for numerical integration
         */
        bool setDiscretizationSamplingTime(const double /*ts*/) override
        {
            throw std::runtime_error("Linear MPC supports only discrete time systems");
            return false;
        }

        /**
         * @brief  Set the solver specific parameters
         *
         * @param param desired parameters (the structure must be of type LParameters)
         */
        void setOptimizerParameters(const Parameters &param) override
        {
            ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setParameters(param);
        }

        /**
         * @brief (NOT AVAILABLE) Set the scaling factor for the control input
         *
         */
        void setInputScale(const cvec<Tnu> /*scaling*/) override
        {
            throw std::runtime_error("Linear MPC does not support input scaling");
        }

        /**
         * @brief (NOT AVAILABLE) Set the scaling factor for the dynamical system's states variables
         *
         */
        void setStateScale(const cvec<Tnx> /*scaling*/) override
        {
            throw std::runtime_error("Linear MPC does not support state scaling");
        }

        /**
         * @brief Sets the bounds for the state variables.
         * 
         * This function sets the lower and upper bounds for the state variables.
         * 
         * @param XMinMat The matrix containing the lower bounds for each state variable.
         * @param XMaxMat The matrix containing the upper bounds for each state variable.
         * @return True if the state bounds were successfully set, false otherwise.
         */
        bool setStateBounds(const mat<Tnx, Tph> &XMinMat, const mat<Tnx, Tph> &XMaxMat) override
        {
            Logger::instance().log(Logger::log_type::DETAIL) << "Setting state bounds" << std::endl;
            return builder.setStateBounds(XMinMat, XMaxMat);
        }

        /**
         * Sets the input bounds for the LMPC controller.
         * 
         * @param UMinMat The matrix representing the lower bounds of the inputs.
         * @param UMaxMat The matrix representing the upper bounds of the inputs.
         * @return True if the input bounds were successfully set, false otherwise.
         */
        bool setInputBounds(const mat<Tnu, Tch> &UMinMat, const mat<Tnu, Tch> &UMaxMat) override
        {
            Logger::instance().log(Logger::log_type::DETAIL) << "Setting input bounds" << std::endl;
            return builder.setInputBounds(UMinMat, UMaxMat);
        }

        /**
         * Sets the output bounds for the LMPC controller.
         *
         * @param YMinMat The matrix of lower bounds for the output variables.
         * @param YMaxMat The matrix of upper bounds for the output variables.
         * @return True if the output bounds were successfully set, false otherwise.
         */
        bool setOutputBounds(const mat<Tny, Tph> &YMinMat, const mat<Tny, Tph> &YMaxMat) override
        {
            Logger::instance().log(Logger::log_type::DETAIL) << "Setting output bounds" << std::endl;
            return builder.setOutputBounds(YMinMat, YMaxMat);
        }

        /**
         * Sets the state bounds for the LMPC (Learning Model Predictive Control) algorithm.
         * 
         * @param XMin The minimum state values for each dimension.
         * @param XMax The maximum state values for each dimension.
         * @param slice An array representing the slice of the prediction horizon to set the state bounds for.
         *              If slice[0] and slice[1] are both -1, the state bounds will be set equally for the entire prediction horizon.
         *              If slice[0] and slice[1] are valid indices, the state bounds will be set only for the specified segment of the prediction horizon.
         * @return True if the state bounds were successfully set, false otherwise.
         */
        bool setStateBounds(const cvec<Tnx> &XMin, const cvec<Tnx> &XMax, const HorizonSlice& slice) override
        {
            // Replicate all along the prediction horizon
            if (isSliceUnset(slice))
            {
                mat<Tnx, Tph> XMinMat, XMaxMat;

                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),XMinMat,nx(), ph());
                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),XMaxMat,nx(), ph());

                for (size_t i = 0; i < ph(); i++)
                {
                    XMinMat.col(i) = XMin;
                    XMaxMat.col(i) = XMax;
                }

                Logger::instance().log(Logger::log_type::DETAIL) << "Setting state bounds equally on the horizon" << std::endl;
                return builder.setStateBounds(XMinMat, XMaxMat);
            }
            else
            {
                // Replicate on segment of the prediction horizon
                if (isPredictionHorizonSliceValid(slice))
                {
                    bool ret = true;

                    for (size_t i = (size_t)slice.start; i < (size_t)slice.end; i++)
                    {
                        Logger::instance().log(Logger::log_type::DETAIL) << "Setting state bounds for the step " << i << std::endl;
                        ret = ret && builder.setStateBounds(i, XMin, XMax);
                    }

                    return ret;
                }
            }

            return false;
        }

        /**
         * Sets the input bounds for the LMPC controller.
         *
         * This function allows you to set the input bounds for the LMPC controller. 
         * The input bounds define the allowable range of values for each input variable at each time step of the prediction horizon.
         *
         * @param UMin The lower bounds for the input variables.
         * @param UMax The upper bounds for the input variables.
         * @param slice An optional parameter that specifies a segment of the prediction horizon to set the input bounds for. 
         * If not provided, the input bounds will be set for the entire prediction horizon.
         *
         * @return Returns true if the input bounds were successfully set, false otherwise.
         */
        bool setInputBounds(const cvec<Tnu> &UMin, const cvec<Tnu> &UMax, const HorizonSlice& slice) override
        {
            // Replicate all along the prediction horizon
            if (isSliceUnset(slice))
            {
                mat<Tnu, Tch> UMinMat, UMaxMat;

                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),UMinMat,nu(), ch());
                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),UMaxMat,nu(), ch());

                for (size_t i = 0; i < ch(); i++)
                {
                    UMinMat.col(i) = UMin;
                    UMaxMat.col(i) = UMax;
                }

                Logger::instance().log(Logger::log_type::DETAIL) << "Setting input bounds equally on the horizon" << std::endl;
                return builder.setInputBounds(UMinMat, UMaxMat);
            }
            else
            {
                // Replicate on segment of the prediction horizon
                if (isControlHorizonSliceValid(slice))
                {
                    bool ret = true;

                    for (size_t i = (size_t)slice.start; i < (size_t)slice.end; i++)
                    {
                        Logger::instance().log(Logger::log_type::DETAIL) << "Setting input bounds for the step " << i << std::endl;
                        ret = ret && builder.setInputBounds(i, UMin, UMax);
                    }

                    return ret;
                }
            }

            return false;
        }

        /**
         * Sets the output bounds for the prediction horizon.
         * 
         * @param YMin The lower bounds for the output variables.
         * @param YMax The upper bounds for the output variables.
         * @param slice An array representing the slice of the prediction horizon to set the bounds for.
         *              If slice[0] and slice[1] are both -1, the bounds will be set for the entire horizon.
         *              Otherwise, the bounds will be set for the segment of the horizon specified by slice[0] and slice[1].
         * @return True if the output bounds were successfully set, false otherwise.
         *         If the slice is out of bounds, an error message will be logged and false will be returned.
         */
        bool setOutputBounds(const cvec<Tny> &YMin, const cvec<Tny> &YMax, const HorizonSlice& slice) override
        {
            // Replicate all along the prediction horizon
            if (isSliceUnset(slice))
            {
                mat<Tny, Tph> YMinMat, YMaxMat;

                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),YMinMat,ny(), ph());
                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),YMaxMat,ny(), ph());

                for (size_t i = 0; i < ph(); i++)
                {
                    YMinMat.col(i) = YMin;
                    YMaxMat.col(i) = YMax;
                }

                Logger::instance().log(Logger::log_type::DETAIL) << "Setting output bounds equally on the horizon" << std::endl;
                return builder.setOutputBounds(YMinMat, YMaxMat);
            }
            else
            {
                // Replicate on segment of the prediction horizon
                if(isPredictionHorizonSliceValid(slice))
                {
                    bool ret = true;

                    for (size_t i = (size_t)slice.start; i < (size_t)slice.end; i++)
                    {
                        Logger::instance().log(Logger::log_type::DETAIL) << "Setting output bounds for the step " << i << std::endl;
                        ret = ret && builder.setOutputBounds(i, YMin, YMax);
                    }

                    return ret;
                }
            }

            return false;
        }

        /**
         * @brief Set the objective function weights, the weights are applied equally
         * along the specified prediction horizon segment
         *
         * @param OWeight weights for the output vector
         * @param UWeight weights for the optimal control input vector
         * @param DeltaUWeight weight for the variation of the optimal control input vector
         * @param slice slice of the prediction horizon step [start end]
         * (if both ends re set to -1 the whole prediction horizon is used)
         * @return true
         * @return false
         */
        bool setObjectiveWeights(
            const mat<Tny, Tph> &OWeightMat,
            const mat<Tnu, Tph> &UWeightMat,
            const mat<Tnu, Tph> &DeltaUWeightMat)
        {
            Logger::instance().log(Logger::log_type::DETAIL) << "Setting weights" << std::endl;
            return builder.setObjective(OWeightMat, UWeightMat, DeltaUWeightMat);
        }

        /**
         * @brief Set the state, input and output box constraints on a specific horizon step
         *
         * @param index the index to apply the constraint
         * @param XMin minimum state vector
         * @param UMin minimum input vector
         * @param YMin minimum output vector
         * @param XMax maximum state vector
         * @param UMax maximum input vector
         * @param YMax maximum output vector
         * @return true
         * @return false
         */
        bool setConstraints(const unsigned int index,
                            const cvec<Tnx> XMin, const cvec<Tnu> UMin, const cvec<Tny> YMin,
                            const cvec<Tnx> XMax, const cvec<Tnu> UMax, const cvec<Tny> YMax)
        {
            if (index >= ph())
            {
                Logger::instance().log(Logger::log_type::ERROR) << "Horizon index out of bounds" << std::endl;
                return false;
            }

            Logger::instance().log(Logger::log_type::DETAIL) << "Setting constraints for the step " << index << std::endl;
            return builder.setConstraints(index, XMin, UMin, YMin, XMax, UMax, YMax);
        }

        /**
         * @brief Set the scalar constraints, the constraints are applied equally
         * along the prediction horizon segment
         *
         * @param Min lower bound
         * @param Max upper bound
         * @param X the vector applied to the state variables
         * @param U the vector applied to the manipulated variables
         * @param slice slice of the prediction horizon step where to apply the constraints [start end]
         * (if both ends re set to -1 the whole prediction horizon is used)
         * @return true
         * @return false
         */
        bool setScalarConstraint(
            const double min, const double max,
            const cvec<Tnx> X, const cvec<Tnu> U,
            const HorizonSlice& slice)
        {
            // Replicate all along the prediction horizon
            if (isSliceUnset(slice))
            {
                // replicate the bounds all along the prediction horizon
                cvec<Tph> Min, Max;

                COND_RESIZE_CVEC(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),Min,ph());
                COND_RESIZE_CVEC(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),Max,ph());

                for (size_t i = 0; i < ph(); i++)
                {
                    Min.row(i) << min;
                    Max.row(i) << max;
                }

                Logger::instance().log(Logger::log_type::DETAIL) << "Setting scalar constraint equally on the horizon" << std::endl;
                return builder.setScalarConstraint(Min, Max, X, U);
            }
            else
            {
                // Replicate on segment of the prediction horizon
                if (isPredictionHorizonSliceValid(slice))
                {
                    bool ret = true;

                    for (size_t i = (size_t)slice.start; i < (size_t)slice.end; i++)
                    {
                        Logger::instance().log(Logger::log_type::DETAIL) << "Setting scalar constraints for the step " << i << std::endl;
                        ret = ret && builder.setScalarConstraint(i, min, max, X, U);
                    }

                    return ret;
                }
            }

            return false;
        }

        /**
         * @brief Set the scalar constraints on a specific horizon step
         *
         * @param index the index to apply the constraint
         * @param min lower bound
         * @param max upper bound
         * @param X the vector applied to the state variables
         * @param U the vector applied to the manipulated variables
         * @return true
         * @return false
         */
        bool setScalarConstraint(
            const unsigned int index,
            const double min, const double max,
            const cvec<Tnx> X, const cvec<Tnu> U)
        {
            if (index >= ph())
            {
                Logger::instance().log(Logger::log_type::ERROR) << "Horizon index out of bounds" << std::endl;
                return false;
            }

            Logger::instance().log(Logger::log_type::DETAIL) << "Setting scalar constraint" << std::endl;
            return builder.setScalarConstraint(index, min, max, X, U);
        }

        /**
         * @brief Set the objective function weights, the weights are applied equally
         * along the specified prediction horizon segment
         *
         * @param OWeight weights for the output vector
         * @param UWeight weights for the optimal control input vector
         * @param DeltaUWeight weight for the variation of the optimal control input vector
         * @param slice slice of the prediction horizon step where to apply the constraints [start end]
         * (if both ends re set to -1 the whole prediction horizon is used)
         * @return true
         * @return false
         */
        bool setObjectiveWeights(
            const cvec<Tny> &OWeight,
            const cvec<Tnu> &UWeight,
            const cvec<Tnu> &DeltaUWeight,
            const HorizonSlice &slice)
        {
            // Replicate all along the prediction horizon
            if (isSliceUnset(slice))
            {
                mat<Tny, Tph> OWeightMat;
                mat<Tnu, Tph> UWeightMat;
                mat<Tnu, Tph> DeltaUWeightMat;

                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),OWeightMat,ny(), ph());
                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),UWeightMat,nu(), ph());
                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),DeltaUWeightMat,nu(), ph());

                for (size_t i = 0; i < ph(); i++)
                {
                    OWeightMat.col(i) = OWeight;
                    UWeightMat.col(i) = UWeight;
                    DeltaUWeightMat.col(i) = DeltaUWeight;
                }

                Logger::instance().log(Logger::log_type::DETAIL) << "Setting weights equally on the horizon" << std::endl;
                return builder.setObjective(OWeightMat, UWeightMat, DeltaUWeightMat);
            }
            else
            {
                // Replicate on segment of the prediction horizon
                if(isPredictionHorizonSliceValid(slice))
                {
                    bool ret = true;

                    for (size_t i = (size_t)slice.start; i < (size_t)slice.end; i++)
                    {
                        Logger::instance().log(Logger::log_type::DETAIL) << "Setting weights for the step " << i << std::endl;
                        ret = ret && builder.setObjective(i, OWeight, UWeight, DeltaUWeight);
                    }

                    return ret;
                }
            }

            return false;
        }

        /**
         * @brief Set the state space model matrices
         * x(k+1) = A*x(k) + B*u(k) + Bd*d(k)
         * y(k) = C*x(k) + Dd*d(k)
         * @param A state update matrix
         * @param B input matrix
         * @param C output matrix
         * @return true
         * @return false
         */
        bool setStateSpaceModel(
            const mat<Tnx, Tnx> &A, const mat<Tnx, Tnu> &B,
            const mat<Tny, Tnx> &C)
        {

            Logger::instance().log(Logger::log_type::DETAIL) << "Setting state space model" << std::endl;
            return builder.setStateModel(A, B, C);
        }

        /**
         * @brief Set the disturbance matrices for the system.
         *
         * This function sets the disturbance matrices for the system of the form:
         *
         * x(k+1) = A*x(k) + B*u(k) + Bd*d(k)
         * y(k) = C*x(k) + Dd*d(k)
         *
         * where Bd is the state disturbance matrix and Dd is the output disturbance matrix.
         *
         * @param Bd The state disturbance matrix of size (Tnx x Tndu).
         * @param Dd The output disturbance matrix of size (Tny x Tndu).
         *
         * @return true if the disturbance matrices were set successfully, false otherwise.
         *
         */
        bool setDisturbances(
            const mat<Tnx, Tndu> &Bd,
            const mat<Tny, Tndu> &Dd)
        {

            Logger::instance().log(Logger::log_type::DETAIL) << "Setting disturbances matrices" << std::endl;
            return builder.setExogenousInput(Bd, Dd);
        }

        /**
         * @brief Set the exogenous inputs vector
         *
         * @param uMeas measured exogenous input
         * @return true
         * @return false
         */
        bool setExogenousInputs(
            const mat<Tndu, Tph> &uMeasMat)
        {
            return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setExogenousInputs(uMeasMat);
        }

        /**
         * @brief Set the exogenous inputs vector, the exogenous inputs are assumed to be constant
         * along the specified prediction horizon segment
         *
         * @param uMeas measured exogenous input
         * @param slice slice of the prediction horizon [start end]
         * (if both ends re set to -1 the whole prediction horizon is used)
         * @return true
         * @return false
         */
        bool setExogenousInputs(
            const cvec<Tndu> &uMeas,
            const HorizonSlice& slice)
        {
            // Replicate all along the control horizon
            if (isSliceUnset(slice))
            {
                mat<Tndu, Tph> uMeasMat;

                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),uMeasMat,ndu(), ph());

                for (size_t i = 0; i < ph(); i++)
                {
                    uMeasMat.col(i) = uMeas;
                }

                return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setExogenousInputs(uMeasMat);
            }
            else
            {
                // Replicate on segment of the control horizon
                if(isControlHorizonSliceValid(slice))
                {
                    bool ret = true;

                    for (size_t i = (size_t)slice.start; i < (size_t)slice.end; i++)
                    {
                        ret = ret && ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setExogenousInputs(i, uMeas);
                    }

                    return ret;
                }
            }

            return false;
        }

        /**
         * @brief Set the references matrix for the objective function
         *
         * @param outRef reference for the output
         * @param cmdRef reference for the optimal control input
         * @param deltaCmdRef reference for the variation of the optimal control input
         * @return true
         * @return false
         */
        bool setReferences(
            const mat<Tny, Tph> outRefMat,
            const mat<Tnu, Tph> cmdRefMat,
            const mat<Tnu, Tph> deltaCmdRefMat)
        {
            return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setReferences(outRefMat, cmdRefMat, deltaCmdRefMat);
        }

        /**
         * @brief Set the references vector for the objective function, the references are assumed to be constant
         * along the specified prediction horizon segment
         *
         * @param outRef reference for the output
         * @param cmdRef reference for the optimal control input
         * @param deltaCmdRef reference for the variation of the optimal control input
         * @param slice slice of the prediction horizon step [start end]
         * (if both ends re set to -1 the whole prediction horizon is used)
         * @return true
         * @return false
         */
        bool setReferences(
            const cvec<Tny> outRef,
            const cvec<Tnu> cmdRef,
            const cvec<Tnu> deltaCmdRef,
            const HorizonSlice& slice)
        {
            // Replicate all along the prediction horizon
            if (isSliceUnset(slice))
            {
                mat<Tny, Tph> outRefMat;
                mat<Tnu, Tph> cmdRefMat;
                mat<Tnu, Tph> deltaCmdRefMat;

                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),outRefMat,ny(), ph());
                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),cmdRefMat,nu(), ph());
                COND_RESIZE_MAT(MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0),deltaCmdRefMat,nu(), ph());

                for (size_t i = 0; i < ph(); i++)
                {
                    outRefMat.col(i) = outRef;
                    cmdRefMat.col(i) = cmdRef;
                    deltaCmdRefMat.col(i) = deltaCmdRef;
                }

                return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setReferences(outRefMat, cmdRefMat, deltaCmdRefMat);
            }
            else
            {
                // Replicate on segment of the prediction horizon
                if(isPredictionHorizonSliceValid(slice))
                {
                    bool ret = true;

                    for (size_t i = (size_t)slice.start; i < (size_t)slice.end; i++)
                    {
                        Logger::instance().log(Logger::log_type::DETAIL) << "Setting references for the step " << i << std::endl;
                        ret = ret && ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setReferences(i, outRef, cmdRef, deltaCmdRef);
                    }

                    return ret;
                }
            }

            return false;
        }

        /**
         * @brief Get the warm start values for the optimizer's primal variables.
         *
         * This function returns the warm start values for the optimizer's primal variables.
         *
         * @return A vector of doubles containing the warm start values for the primal variables.
         *
         * @note This function assumes that the optimizer has already been run and that the
         *       warm start values for the primal variables have been computed and stored in
         *       the optimizer. If the optimizer has not been run or the warm start values
         *       for the primal variables have not been computed, this function may not return
         *       a valid result.
         *
         * @see LOptimizer, MPCSize
         */
        std::vector<double> getSolverWarmStartPrimal()
        {
            return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->optimal_prev_x;
        }

        /**
         * @brief Get the warm start values for the optimizer's dual variables.
         *
         * This function returns the warm start values for the optimizer's dual variables.
         *
         * @return A vector of doubles containing the warm start values for the dual variables.
         *
         * @note This function assumes that the optimizer has already been run and that the
         *       warm start values for the dual variables have been computed and stored in
         *       the optimizer. If the optimizer has not been run or the warm start values
         *       for the dual variables have not been computed, this function may not return
         *       a valid result.
         *
         * @see LOptimizer, MPCSize
         */
        std::vector<double> getSolverWarmStartDual()
        {
            return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->optimal_prev_y;
        }

        /**
         * @brief Set the warm start values for the optimizer.
         *
         * This function sets the warm start values for the optimizer's primal and dual variables.
         *
         * @param warm_primal A vector of doubles containing the warm start values for the primal variables.
         * @param warm_dual A vector of doubles containing the warm start values for the dual variables.
         *
         * @note This function assumes that the optimizer has already been initialized with the appropriate
         *       problem size and structure. If the optimizer has not been initialized, this function may
         *       not set the warm start values correctly.
         *
         * @see LOptimizer, MPCSize
         */
        void setSolverWarmStart(std::vector<double> warm_primal, std::vector<double> warm_dual)
        {
            auto *optimizer = dynamic_cast<LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *>(optPtr);

            optimizer->optimal_prev_x = warm_primal;
            optimizer->optimal_prev_y = warm_dual;
        }

    protected:
        /**
         * @brief Initilization hook for the linear interface
         */
        void onSetup() override
        {
            builder.initialize(nx(), nu(), ndu(), ny(), ph(), ch());
            optPtr = new LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>();
            optPtr->initialize(nx(), nu(), ndu(), ny(), ph(), ch());

            ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setBuilder(&builder);
        }

        /**
         * @brief This function is a hook that is called when the dynamical system initial condition is updated.
         *
         * This function does not perform any action and is not available for use.
         *
         * @param x0 The updated initial condition of the dynamical system.
         *
         * @warning This function is not available for use.
         */
        void onModelUpdate(const cvec<Tnx> /*x0*/) override
        {
        }

    private:
        ProblemBuilder<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> builder;
    };
} // namespace mpc
