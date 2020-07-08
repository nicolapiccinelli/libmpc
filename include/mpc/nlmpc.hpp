#pragma once

#include <mpc/conFunction.hpp>
#include <mpc/objFunction.hpp>
#include <mpc/optimizer.hpp>

#include <chrono>

namespace mpc
{

    template <
        int Tnx = Eigen::Dynamic, int Tnu = Eigen::Dynamic, int Tny = Eigen::Dynamic,
        int Tph = Eigen::Dynamic, int Tch = Eigen::Dynamic,
        int Tineq = Eigen::Dynamic, int Teq = Eigen::Dynamic>
    class NLMPC : public Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>
    {
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_initialize;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_checkOrQuit;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::GetSize;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_dimensions;

    public:
        NLMPC() = default;

        void initialize(
            bool hardConstraints,
            int tnx = Tnx, int tnu = Tnu, int tny = Tny,
            int tph = Tph, int tch = Tch,
            int tineq = Tineq, int teq = Teq)
        {
            _initialize(tnx, tnu, tny, tph, tch, tineq, teq);

            _conFunc.initialize(_dimensions.tnx, _dimensions.tnu, _dimensions.tny, _dimensions.tph, _dimensions.tch, _dimensions.tineq, _dimensions.teq);
            _mapping.initialize(_dimensions.tnx, _dimensions.tnu, _dimensions.tny, _dimensions.tph, _dimensions.tch, _dimensions.tineq, _dimensions.teq);
            _objFunc.initialize(_dimensions.tnx, _dimensions.tnu, _dimensions.tny, _dimensions.tph, _dimensions.tch, _dimensions.tineq, _dimensions.teq);

            _objFunc.setMapping(_mapping);
            _conFunc.setMapping(_mapping);

            _opt = new Optimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
            _opt->initialize(hardConstraints, _dimensions.tnx, _dimensions.tnu, _dimensions.tny, _dimensions.tph, _dimensions.tch, _dimensions.tineq, _dimensions.teq);
            _opt->setMapping(_mapping);

            _result.cmd.resize(_dimensions.tnu);
            _result.cmd.setZero();

            Logger::instance().log(Logger::log_type::INFO) 
                << "Mapping assignment done" 
                << std::endl;
        }

        ~NLMPC()
        {
            _checkOrQuit();

            delete _opt;
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
            _checkOrQuit();

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Setting sampling time to: " 
                << ts 
                << " sec(s)" 
                << std::endl;

            auto res = _conFunc.setContinuos(true, ts);
            return res;
        } // Can this be initialized at construction?

        void setTolerances(const Parameters param)
        {
            _checkOrQuit();
            _opt->setTolerances(param);
        }

        bool setObjectiveFunction(const typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::ObjFunHandle handle)
        {
            _checkOrQuit();

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Setting objective function handle" 
                << std::endl;

            auto res = _objFunc.setUserFunction(handle);

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Binding objective function handle" 
                << std::endl;

            _opt->bind(&_objFunc);
            return res;
        }

        bool setStateSpaceFunction(const typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::StateFunHandle handle)
        {
            _checkOrQuit();

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Setting state space function handle" 
                << std::endl;

            auto res = _conFunc.setStateSpaceFunction(handle);

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Binding state space constraints" 
                << std::endl;

            static cvec<AssignSize(sizeEnum::StateIneqSize)> ineq_tol;
            ineq_tol.resize(GetSize(sizeEnum::StateIneqSize));
            ineq_tol.setOnes();
            
            _opt->bindIneq(&_conFunc, constraints_type::INEQ, ineq_tol * 1e-10);

            static cvec<AssignSize(sizeEnum::StateEqSize)> eq_tol;
            eq_tol.resize(GetSize(sizeEnum::StateEqSize));
            eq_tol.setOnes();

            _opt->bindEq(&_conFunc, constraints_type::EQ, eq_tol * 1e-10);

            return res;
        }

        bool setIneqConFunction(
            const typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle handle)
        {
            static cvec<Tineq> tol;
            tol = cvec<Tineq>::Ones(_dimensions.tineq) * 1e-10;
            return setIneqConFunction(handle, tol);            
        }

        bool setIneqConFunction(
            const typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle handle,
            const cvec<Tineq> tol)
        {
            _checkOrQuit();

            auto res = _conFunc.setIneqConstraintFunction(handle);
            _opt->bindUserIneq(&_conFunc, constraints_type::UINEQ, tol);
            return res;
        }

        bool setEqConFunction(
            const typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle handle)
        {
            static cvec<Teq> tol;
            tol = cvec<Tineq>::Ones(_dimensions.teq) * 1e-10;
            return setEqConFunction(handle, tol);
        }

        bool setEqConFunction(
            const typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle handle,
            const cvec<Teq> tol)
        {
            _checkOrQuit();

            auto res = _conFunc.setEqConstraintFunction(handle);
            _opt->bindUserEq(&_conFunc, constraints_type::UEQ, tol);
            return res;
        }

        bool setOutputFunction(const typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::OutFunHandle handle)
        {
            _checkOrQuit();

            return _conFunc.setOutputFunction(handle);
        }

        typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::Result step(
            const cvec<Tnx> x0,
            const cvec<Tnu> lastU)
        {
            _checkOrQuit();

            _objFunc.setCurrentState(x0);
            _conFunc.setCurrentState(x0);

            Logger::instance().log(Logger::log_type::INFO) 
                << "Optimization step" 
                << std::endl;

            auto start = std::chrono::steady_clock::now();
            _result = _opt->run(x0, lastU);
            auto stop = std::chrono::steady_clock::now();

            Logger::instance().log(Logger::log_type::INFO) 
                << "Optimization step duration: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() 
                << " (ms)" 
                << std::endl;
            return _result;
        }

        typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::Result getLastResult()
        {
            _checkOrQuit();

            return _result;
        }

    protected:
        ObjFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> _objFunc;
        ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> _conFunc;
        Optimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> *_opt;     
        Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> _mapping;    

        typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::Result _result;
    };

} // namespace mpc
