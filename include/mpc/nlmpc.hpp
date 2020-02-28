#pragma once

#include <mpc/conFunction.hpp>
#include <mpc/mpc.hpp>
#include <mpc/objFunction.hpp>
#include <mpc/optimizer.hpp>

#include <chrono>

namespace mpc {
template <std::size_t Tnx, std::size_t Tnu, std::size_t Tny, std::size_t Tph, std::size_t Tch, std::size_t Tineq, std::size_t Teq>
class NLMPC {
public:
    NLMPC() = default;
    NLMPC(bool hardConstraints, bool verbose = false)
    {
        init(hardConstraints, verbose);
    }

    void init(bool hardConstraints, bool verbose = false)
    {
        Logger::verbose = (int)verbose;
        dbg(Logger::INFO) << "Verbosity mode active" << std::endl;

        objFunc.setMapping(mapping);
        conFunc.setMapping(mapping);

        opt = new Optimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>(hardConstraints);
        opt->setMapping(mapping);

        dbg(Logger::INFO) << "Mapping assignment done" << std::endl;
    }

    ~NLMPC()
    {
        delete opt;
    };

    bool setVerbosity(bool status, Logger::level l = Logger::INFO)
    {
        Logger::verbose = (int)status;
        Logger::logLevel = l;
        return true;
    }

    bool setLoggerPrefix(std::string prefix)
    {
        Logger::prefix = prefix;
    }

    bool setSampleTime(const double ts)
    {
        dbg(Logger::DEEP) << "Setting sampling time to: " << ts << " sec(s)" << std::endl;

        auto res = conFunc.setContinuos(true, ts);
        return res;
    }

    void setTolerances(Parameters param)
    {
        opt->setTolerances(param);
    }

    bool setObjectiveFunction(const ObjFunHandle<Tph, Tnx, Tnu> handle)
    {
        dbg(Logger::DEEP) << "Setting objective function handle" << std::endl;

        auto res = objFunc.setUserFunction(handle);

        dbg(Logger::DEEP) << "Binding objective function handle" << std::endl;

        opt->bind(&objFunc);
        return res;
    }

    bool setStateSpaceFunction(const StateFunHandle<Tnx, Tnu> handle)
    {
        dbg(Logger::DEEP) << "Setting state space function handle" << std::endl;

        auto res = conFunc.setStateSpaceFunction(handle);

        cvec<StateIneqSize> ineq_tol;
        ineq_tol.setOnes();
        opt->bindIneq(&conFunc, constraints_type::INEQ, ineq_tol * 1e-10);

        cvec<StateEqSize> eq_tol;
        eq_tol.setOnes();
        opt->bindEq(&conFunc, constraints_type::EQ, eq_tol * 1e-10);

        dbg(Logger::DEEP) << "Binding state space constraints" << std::endl;

        return res;
    }

    bool setIneqConFunction(const IConFunHandle<Tineq, Tph, Tnx, Tnu> handle, const cvec<Tineq> tol = cvec<Tineq>::Ones() * 1e-10)
    {
        auto res = conFunc.setIneqConstraintFunction(handle);
        opt->bindUserIneq(&conFunc, constraints_type::UINEQ, tol);
        return res;
    }

    bool setEqConFunction(const EConFunHandle<Teq, Tph, Tnx, Tnu> handle, const cvec<Teq> tol = cvec<Teq>::Ones() * 1e-10)
    {
        auto res = conFunc.setEqConstraintFunction(handle);
        opt->bindUserEq(&conFunc, constraints_type::UEQ, tol);
        return res;
    }

    bool setOutputFunction(const OutFunHandle handle)
    {
        return conFunc.setOutputFunction(handle);
    }

    Result<Tnu> step(const cvec<Tnx> x0, const cvec<Tnu> lastU)
    {
        objFunc.setCurrentState(x0);
        conFunc.setCurrentState(x0);

        dbg(Logger::INFO) << "Optimization step" << std::endl;

        auto start = std::chrono::steady_clock::now();
        Result<Tnu> res = opt->run(x0, lastU);
        auto stop = std::chrono::steady_clock::now();

        dbg(Logger::INFO) << "Optimization step duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " (ms)" << std::endl;
        return res;
    }

private:
    double sampleTs;

    ObjFunction<Tnx, Tnu, Tph, Tch> objFunc;
    ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conFunc;
    Optimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* opt;
    Common<Tnx, Tnu, Tph, Tch> mapping;
};
} // namespace mpc