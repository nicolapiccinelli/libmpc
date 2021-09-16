#pragma once

#include <mpc/IMPC.hpp>
#include <mpc/LOptimizer.hpp>
#include <mpc/ProblemBuilder.hpp>

namespace mpc {
template <
    int Tnx = Eigen::Dynamic, int Tnu = Eigen::Dynamic, int Tndu = Eigen::Dynamic,
    int Tny = Eigen::Dynamic, int Tph = Eigen::Dynamic, int Tch = Eigen::Dynamic>
class LMPC : public IMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0> {
private:
    using IMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::optPtr;
    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::dim;

public:
    using IMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::step;
    using IMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::setLoggerLevel;
    using IMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::setLoggerPrefix;
    using IMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::getLastResult;

public:
    LMPC() = default;
    ~LMPC() = default;

    bool setContinuosTimeModel(const double /*ts*/)
    {
        throw std::runtime_error("Linear MPC supports only discrete time systems");
        return false;
    }

    void setOptimizerParameters(const Parameters param)
    {
        checkOrQuit();
        ((LOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch>*)optPtr)->setParameters(param);
    }

    void setInputScale(const cvec<Tnu> /*scaling*/)
    {
        throw std::runtime_error("Linear MPC does not support input scaling");
        checkOrQuit();
    }

    void setStateScale(const cvec<Tnx> /*scaling*/)
    {
        throw std::runtime_error("Linear MPC does not support state scaling");
        checkOrQuit();
    }

    bool setConstraints(
        const cvec<Tnx> XMin, const cvec<Tnu> UMin, const cvec<Tny> YMin,
        const cvec<Tnx> XMax, const cvec<Tnu> UMax, const cvec<Tny> YMax)
    {
        checkOrQuit();

        // replicate the bounds all along the prediction horizon
        mat<Tnx, Tph> XMinMat, XMaxMat;
        mat<Tny, Tph> YMinMat, YMaxMat;
        mat<Tnu, Tph> UMinMat, UMaxMat;

        XMinMat.resize(dim.nx.num(), dim.ph.num());
        YMinMat.resize(dim.ny.num(), dim.ph.num());
        UMinMat.resize(dim.nu.num(), dim.ph.num());

        XMaxMat.resize(dim.nx.num(), dim.ph.num());
        YMaxMat.resize(dim.ny.num(), dim.ph.num());
        UMaxMat.resize(dim.nu.num(), dim.ph.num());

        for (size_t i = 0; i < dim.ph.num(); i++) {
            XMinMat.col(i) = XMin;
            XMaxMat.col(i) = XMax;
            YMinMat.col(i) = YMin;
            YMaxMat.col(i) = YMax;

            if (i < dim.ph.num()) {
                UMinMat.col(i) = UMin;
                UMaxMat.col(i) = UMax;
            }
        }

        Logger::instance().log(Logger::log_type::DETAIL) << "Setting constraints" << std::endl;
        return builder.setConstraints(
            XMinMat, UMinMat, YMinMat,
            XMaxMat, UMaxMat, YMaxMat);
    }

    bool setObjectiveWeights(
        const cvec<Tny>& OWeight,
        const cvec<Tnu>& UWeight,
        const cvec<Tnu>& DeltaUWeight)
    {
        checkOrQuit();

        // replicate the weights all along the prediction horizon
        mat<Tny, (dim.ph + Dim<1>())> OWeightMat;
        mat<Tnu, (dim.ph + Dim<1>())> UWeightMat;
        mat<Tnu, Tph> DeltaUWeightMat;

        OWeightMat.resize(dim.ny.num(), dim.ph.num() + 1);
        UWeightMat.resize(dim.nu.num(), dim.ph.num() + 1);
        DeltaUWeightMat.resize(dim.nu.num(), dim.ph.num());

        for (size_t i = 0; i < dim.ph.num() + 1; i++) {
            OWeightMat.col(i) = OWeight;
            UWeightMat.col(i) = UWeight;
            if (i < dim.ph.num()) {
                DeltaUWeightMat.col(i) = DeltaUWeight;
            }
        }

        Logger::instance().log(Logger::log_type::DETAIL) << "Setting weights" << std::endl;
        return builder.setObjective(OWeightMat, UWeightMat, DeltaUWeightMat);
    }

    bool setStateSpaceModel(
        const mat<Tnx, Tnx>& A, const mat<Tnx, Tnu>& B,
        const mat<Tny, Tnx>& C)
    {
        checkOrQuit();

        Logger::instance().log(Logger::log_type::DETAIL) << "Setting state space model" << std::endl;
        return builder.setStateModel(A, B, C);
    }

    bool setDisturbances(
        const mat<Tnx, Tndu> &B, 
        const mat<Tny, Tndu> &D
    )
    {
        checkOrQuit();

        Logger::instance().log(Logger::log_type::DETAIL) << "Setting disturbances matrices" << std::endl;
        return builder.setExogenuosInput(B, D);
    }

    bool setExogenuosInputs(
        const cvec<Tndu>& uMeas)
    {
        return ((LOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch>*)optPtr)->setExogenuosInputs(uMeas);
    }

    bool setReferences(
        const cvec<Tny> outRef,
        const cvec<Tnu> cmdRef,
        const cvec<Tnu> deltaCmdRef)
    {
        return ((LOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch>*)optPtr)->setReferences(outRef, cmdRef, deltaCmdRef);
    }

protected:
    void onSetup()
    {
        builder.initialize(
            dim.nx.num(), dim.nu.num(), dim.ndu.num(), dim.ny.num(),
            dim.ph.num(), dim.ch.num());

        optPtr = new LOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch>();
        optPtr->initialize(
            dim.nx.num(), dim.nu.num(), dim.ndu.num(), dim.ny.num(),
            dim.ph.num(), dim.ch.num());

        ((LOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch>*)optPtr)->setBuilder(&builder);
    }

    void onModelUpdate(const cvec<Tnx> /*x0*/)
    {

    }

private:
    ProblemBuilder<Tnx, Tnu, Tndu, Tny, Tph, Tch> builder;

    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::checkOrQuit;
    using IMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::result;
};

} // namespace mpc
