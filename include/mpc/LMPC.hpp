#pragma once

#include <mpc/IMPC.hpp>
#include <mpc/LOptimizer.hpp>
#include <mpc/ProblemBuilder.hpp>

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
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::step;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::setLoggerLevel;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::setLoggerPrefix;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::getLastResult;

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
        bool setContinuosTimeModel(const double /*ts*/)
        {
            throw std::runtime_error("Linear MPC supports only discrete time systems");
            return false;
        }

        /**
         * @brief  Set the solver specific parameters
         *
         * @param param desired parameters (the structure must be of type LParameters)
         */
        void setOptimizerParameters(const Parameters &param)
        {
            ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setParameters(param);
        }

        /**
         * @brief (NOT AVAILABLE) Set the scaling factor for the control input
         *
         */
        void setInputScale(const cvec<Tnu> /*scaling*/)
        {
            throw std::runtime_error("Linear MPC does not support input scaling");
        }

        /**
         * @brief (NOT AVAILABLE) Set the scaling factor for the dynamical system's states variables
         *
         */
        void setStateScale(const cvec<Tnx> /*scaling*/)
        {
            throw std::runtime_error("Linear MPC does not support state scaling");
        }

        /**
         * @brief Set the state, input and output box constraints, the constraints are applied equally
         * along the prediction horizon
         *
         * @param XMin minimum state vector
         * @param UMin minimum input vector
         * @param YMin minimum output vector
         * @param XMax maximum state vector
         * @param UMax maximum input vector
         * @param YMax maximum output vector
         * @return true
         * @return false
         */
        bool setConstraints(
            const cvec<Tnx> XMin, const cvec<Tnu> UMin, const cvec<Tny> YMin,
            const cvec<Tnx> XMax, const cvec<Tnu> UMax, const cvec<Tny> YMax)
        {

            // replicate the bounds all along the prediction horizon
            mat<Tnx, Tph> XMinMat, XMaxMat;
            mat<Tny, Tph> YMinMat, YMaxMat;
            mat<Tnu, Tph> UMinMat, UMaxMat;

            XMinMat.resize(nx(), ph());
            YMinMat.resize(ny(), ph());
            UMinMat.resize(nu(), ph());

            XMaxMat.resize(nx(), ph());
            YMaxMat.resize(ny(), ph());
            UMaxMat.resize(nu(), ph());

            for (size_t i = 0; i < ph(); i++)
            {
                XMinMat.col(i) = XMin;
                XMaxMat.col(i) = XMax;
                YMinMat.col(i) = YMin;
                YMaxMat.col(i) = YMax;

                if (i < ph())
                {
                    UMinMat.col(i) = UMin;
                    UMaxMat.col(i) = UMax;
                }
            }

            Logger::instance().log(Logger::log_type::DETAIL) << "Setting constraints" << std::endl;
            return builder.setConstraints(
                XMinMat, UMinMat, YMinMat,
                XMaxMat, UMaxMat, YMaxMat);
        }

        /**
         * @brief Set the objective function weights, the weights are applied equally
         * along the prediction horizon
         *
         * @param OWeight weights for the output vector
         * @param UWeight weights for the optimal control input vector
         * @param DeltaUWeight weight for the variation of the optimal control input vector
         * @return true
         * @return false
         */
        bool setObjectiveWeights(
            const cvec<Tny> &OWeight,
            const cvec<Tnu> &UWeight,
            const cvec<Tnu> &DeltaUWeight)
        {
            // replicate the weights all along the prediction horizon
            mat<Tny, (Size(Tph) + 1)> OWeightMat;
            mat<Tnu, (Size(Tph) + 1)> UWeightMat;
            mat<Tnu, Tph> DeltaUWeightMat;

            OWeightMat.resize(ny(), ph() + 1);
            UWeightMat.resize(nu(), ph() + 1);
            DeltaUWeightMat.resize(nu(), ph());

            for (size_t i = 0; i < ph() + 1; i++)
            {
                OWeightMat.col(i) = OWeight;
                UWeightMat.col(i) = UWeight;
                if (i < ph())
                {
                    DeltaUWeightMat.col(i) = DeltaUWeight;
                }
            }

            Logger::instance().log(Logger::log_type::DETAIL) << "Setting weights" << std::endl;
            return builder.setObjective(OWeightMat, UWeightMat, DeltaUWeightMat);
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
         * @brief Set the disturbances matrices
         * x(k+1) = A*x(k) + B*u(k) + Bd*d(k)
         * y(k) = C*x(k) + Dd*d(k)
         * @param Bd state disturbance matrix
         * @param Dd output disturbance matrix
         * @return true
         * @return false
         */
        bool setDisturbances(
            const mat<Tnx, Tndu> &Bd,
            const mat<Tny, Tndu> &Dd)
        {

            Logger::instance().log(Logger::log_type::DETAIL) << "Setting disturbances matrices" << std::endl;
            return builder.setExogenuosInput(Bd, Dd);
        }

        /**
         * @brief Set the exogenuos inputs vector
         *
         * @param uMeas measured exogenuos input
         * @return true
         * @return false
         */
        bool setExogenuosInputs(
            const cvec<Tndu> &uMeas)
        {
            return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setExogenuosInputs(uMeas);
        }

        /**
         * @brief Set the references vector for the objective function
         *
         * @param outRef reference for the output
         * @param cmdRef reference for the optimal control input
         * @param deltaCmdRef reference for the variation of the optimal control input
         * @return true
         * @return false
         */
        bool setReferences(
            const cvec<Tny> outRef,
            const cvec<Tnu> cmdRef,
            const cvec<Tnu> deltaCmdRef)
        {
            return ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setReferences(outRef, cmdRef, deltaCmdRef);
        }

    protected:
        /**
         * @brief Initilization hook for the linear interface
         */
        void onSetup()
        {
            builder.initialize(
                nx(), nu(), ndu(), ny(),
                ph(), ch());

            optPtr = new LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>();
            optPtr->initialize(
                nx(), nu(), ndu(), ny(),
                ph(), ch());

            ((LOptimizer<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> *)optPtr)->setBuilder(&builder);
        }

        /**
         * @brief (NOT AVAILABLE) Dynamical system initial condition update hook
         */
        void onModelUpdate(const cvec<Tnx> /*x0*/)
        {
        }

    private:
        ProblemBuilder<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)> builder;
        using IMPC<MPCSize(Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0)>::result;
    };
} // namespace mpc
