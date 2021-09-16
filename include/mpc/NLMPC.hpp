#pragma once

#include <mpc/IMPC.hpp>

#include <mpc/Constraints.hpp>
#include <mpc/Objective.hpp>
#include <mpc/NLOptimizer.hpp>

namespace mpc {
template <
    int Tnx = Eigen::Dynamic, int Tnu = Eigen::Dynamic, int Tny = Eigen::Dynamic,
    int Tph = Eigen::Dynamic, int Tch = Eigen::Dynamic,
    int Tineq = Eigen::Dynamic, int Teq = Eigen::Dynamic>
class NLMPC : public IMPC<Tnx, Tnu,0, Tny, Tph, Tch, Tineq, Teq> {
private:
    using Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::checkOrQuit;
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::optPtr;
    using Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::dim;

public:
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::step;
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::setLoggerLevel;
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::setLoggerPrefix;
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::getLastResult;

    NLMPC() = default;

    ~NLMPC()
    {
        checkOrQuit();
        delete optPtr;
    }

    bool setContinuosTimeModel(const double ts)
    {
        checkOrQuit();

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting sampling time to: "
            << ts
            << " sec(s)"
            << std::endl;

        auto res = conF.setContinuos(true, ts);
        return res;
    }

    void setOptimizerParameters(const Parameters param)
    {
        checkOrQuit();
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setParameters(param);
    }

    void setInputScale(const cvec<Tnu> scaling)
    {
        mapping.setInputScaling(scaling);

        objF.setMapping(mapping);
        conF.setMapping(mapping);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setMapping(mapping);
    }

    void setStateScale(const cvec<Tnx> scaling)
    {
        mapping.setStateScaling(scaling);

        objF.setMapping(mapping);
        conF.setMapping(mapping);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setMapping(mapping);
    }

    bool setObjectiveFunction(const typename Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::ObjFunHandle handle)
    {
        checkOrQuit();

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting objective function handle"
            << std::endl;

        auto res = objF.setObjective(handle);

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Binding objective function handle"
            << std::endl;

        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bind(&objF);
        return res;
    }

    bool setStateSpaceFunction(const typename Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::StateFunHandle handle,
        const float eq_tol = 1e-10)
    {
        checkOrQuit();

        static cvec<Dim<2>() * dim.ph * dim.ny> ineq_tol_vec;
        ineq_tol_vec.resize((2 * dim.ph.num() * dim.ny.num()));
        ineq_tol_vec.setOnes();

        static cvec<(dim.ph * dim.nx)> eq_tol_vec;
        eq_tol_vec.resize((dim.ph.num() * dim.nx.num()));
        eq_tol_vec.setOnes();

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting state space function handle"
            << std::endl;

        bool res = conF.setStateModel(handle);

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Binding state space constraints"
            << std::endl;

        res = res & ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bindEq(&conF, constraints_type::EQ, eq_tol_vec * eq_tol);

        return res;
    }

    bool setIneqConFunction(
        const typename Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle handle, const float tol = 1e-10)
    {
        checkOrQuit();

        cvec<Tineq> tol_vec;
        tol_vec = cvec<Tineq>::Ones(dim.ineq.num());

        auto res = conF.setIneqConstraints(handle);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bindUserIneq(&conF, constraints_type::UINEQ, tol_vec * tol);
        return res;
    }

    bool setEqConFunction(
        const typename Common<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle handle, const float tol = 1e-10)
    {
        checkOrQuit();

        cvec<Teq> tol_vec;
        tol_vec = cvec<Teq>::Ones(dim.eq);

        auto res = conF.setEqConstraints(handle);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bindUserEq(&conF, constraints_type::UEQ, tol_vec * tol);
        return res;
    }

protected:
    void onSetup()
    {
        conF.initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        mapping.initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        objF.initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        optPtr = new NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
        optPtr->initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        objF.setMapping(mapping);
        conF.setMapping(mapping);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setMapping(mapping);

        Logger::instance().log(Logger::log_type::INFO)
            << "Mapping assignment done"
            << std::endl;
    }

    void onModelUpdate(const cvec<Tnx> x0)
    {
        objF.setCurrentState(x0);
        conF.setCurrentState(x0);
    }

private:
    Objective<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> objF;
    Constraints<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conF;
    Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> mapping;

    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::result;
};

} // namespace mpc
