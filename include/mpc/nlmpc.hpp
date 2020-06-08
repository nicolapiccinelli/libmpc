#pragma once

#include <mpc/conFunction.hpp>
#include <mpc/mpc.hpp>
#include <mpc/objFunction.hpp>
#include <mpc/optimizer.hpp>

#include <chrono>

namespace mpc {
template <std::size_t Tnx,
          std::size_t Tnu,
          std::size_t Tny,
          std::size_t Tph,
          std::size_t Tch,
          std::size_t Tineq,
          std::size_t Teq>
class NLMPC {
public:
    NLMPC()
    {
        opt = NULL;
    };

    void initialize(bool hardConstraints)
    {       
        // Why can't those be initialized in ObjFunction and ConFunction?
        objFunc.setMapping(mapping);
        conFunc.setMapping(mapping);

        // Do we really need to be able to change HardConstraints at runtime?
        opt = new Optimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>(hardConstraints);        
        opt->setMapping(mapping);

        Logger::instance().log(Logger::log_type::INFO) << "Mapping assignment done" << std::endl;
    } // Can this be initialized at construction?

    ~NLMPC()
    {
        if(opt != NULL)
            delete opt;
    };

    bool setLoggerLevel(Logger::log_level l)
    {
        Logger::instance().setLevel(l);
        return true;
    }

    bool setLoggerPrefix(std::string prefix)
    {
        Logger::instance().setPrefix(prefix);
        return true;
    }

    bool setSampleTime(const double ts)
    {
        Logger::instance().log(Logger::log_type::DEBUG) << "Setting sampling time to: " << ts << " sec(s)" << std::endl;

        auto res = conFunc.setContinuos(true, ts);
        return res;
    } // Can this be initialized at construction?

    void setTolerances(Parameters param)
    {
        opt->setTolerances(param);
    }

    bool setObjectiveFunction(const ObjFunHandle<Tph, Tnx, Tnu> handle)
    {
        checkOrQuit();

        Logger::instance().log(Logger::log_type::DEBUG) << "Setting objective function handle" << std::endl;

        auto res = objFunc.setUserFunction(handle);

        Logger::instance().log(Logger::log_type::DEBUG) << "Binding objective function handle" << std::endl;

        opt->bind(&objFunc);
        return res;
    }

    bool setStateSpaceFunction(const StateFunHandle<Tnx, Tnu> handle)
    {
        checkOrQuit();

        Logger::instance().log(Logger::log_type::DEBUG) << "Setting state space function handle" << std::endl;

        auto res = conFunc.setStateSpaceFunction(handle);

        Logger::instance().log(Logger::log_type::DEBUG) << "Binding state space constraints" << std::endl;

        cvec<StateIneqSize> ineq_tol;
        ineq_tol.setOnes();
        opt->bindIneq(&conFunc, constraints_type::INEQ, ineq_tol * 1e-10);

        cvec<StateEqSize> eq_tol;
        eq_tol.setOnes();
        opt->bindEq(&conFunc, constraints_type::EQ, eq_tol * 1e-10);

        return res;
    }

    bool setIneqConFunction(const IConFunHandle<Tineq, Tph, Tnx, Tnu> handle, const cvec<Tineq> tol = cvec<Tineq>::Ones() * 1e-10)
    {
        checkOrQuit();

        auto res = conFunc.setIneqConstraintFunction(handle);
        opt->bindUserIneq(&conFunc, constraints_type::UINEQ, tol);
        return res;
    }

    bool setEqConFunction(const EConFunHandle<Teq, Tph, Tnx, Tnu> handle, const cvec<Teq> tol = cvec<Teq>::Ones() * 1e-10)
    {
        checkOrQuit();

        auto res = conFunc.setEqConstraintFunction(handle);
        opt->bindUserEq(&conFunc, constraints_type::UEQ, tol);
        return res;
    }

    bool setOutputFunction(const OutFunHandle handle)
    {
        checkOrQuit();

        return conFunc.setOutputFunction(handle);
    }

    Result<Tnu> step(const cvec<Tnx> x0, const cvec<Tnu> lastU)
    {
        checkOrQuit();

        objFunc.setCurrentState(x0);
        conFunc.setCurrentState(x0);

        Logger::instance().log(Logger::log_type::INFO) << "Optimization step" << std::endl;

        auto start = std::chrono::steady_clock::now();
        Result<Tnu> res = opt->run(x0, lastU);
        auto stop = std::chrono::steady_clock::now();

        Logger::instance().log(Logger::log_type::INFO) << "Optimization step duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " (ms)" << std::endl;
        return res;
    }

private:

    inline void checkOrQuit(){
        if (!opt){
            Logger::instance().log(Logger::log_type::ERROR) << RED << "MPC is not initialized, quitting..." << RESET << std::endl;
            exit(-1);
        }
    }

    ObjFunction<Tnx, Tnu, Tph, Tch> objFunc;
    ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conFunc;
    Optimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* opt;
    Common<Tnx, Tnu, Tph, Tch> mapping;
};
} // namespace mpc
