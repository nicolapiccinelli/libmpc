#pragma once

#include <mpc/mapping.hpp>
#include <mpc/types.hpp>
#include <mpc/conFunction.hpp>
#include <mpc/objFunction.hpp>
#include <mpc/logger.hpp>
#include <nlopt.hpp>

namespace mpc {
template <
        int Tnx,
        int Tnu,
        int Tny,
        int Tph,
        int Tch,
        int Tineq,
        int Teq>
class Optimizer :
    public Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>
{
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_initialize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_checkOrQuit;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::GetSize;

    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_dimensions;

public:
    Optimizer() : Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>() {};

    void initialize(
        bool hardConstraints,
        int tnx, int tnu, int tny,
        int tph, int tch,
        int tineq, int teq)
    {
        _initialize(tnx, tnu, tny, tph, tch, tineq, teq);
        _hard = hardConstraints;

        _innerOpt = new nlopt::opt(nlopt::LD_SLSQP, GetSize(sizeEnum::DecVarsSize));
        setDefaultBounds();

        Parameters p;
        setTolerances(p);

        _last_r.cmd.resize(_dimensions.tnu);
        _last_r.cmd.setZero();

        _currentSlack = 0;
    }

    ~Optimizer()
    {
        _checkOrQuit();

        delete _innerOpt;
    }

    void setMapping(Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>& m)
    {
        _checkOrQuit();

        _mapping = m;
    }

    void setDefaultBounds()
    {
        _checkOrQuit();

        static std::vector<double> lb, ub;
        lb.resize(GetSize(sizeEnum::DecVarsSize));
        ub.resize(GetSize(sizeEnum::DecVarsSize));

        for (int i = 0; i < GetSize(sizeEnum::DecVarsSize); i++)
		{
            lb[i] = -std::numeric_limits<double>::infinity();
            if (i + 1 == GetSize(sizeEnum::DecVarsSize) && _hard)
            {
                lb[i] = 0;
            }
            ub[i] = std::numeric_limits<double>::infinity();
        }

        _innerOpt->set_lower_bounds(lb);
        _innerOpt->set_upper_bounds(ub);

        // TODO support output bounds
        _outputBounds = false;
    }

    void setTolerances(Parameters param)
    {
        _checkOrQuit();

        _innerOpt->set_ftol_rel(param.relative_ftol);
        _innerOpt->set_maxeval(param.maximum_iteration);
        _innerOpt->set_xtol_rel(param.relative_xtol);

        Logger::instance().log(Logger::log_type::DEBUG) 
            << "Setting tolerances and stopping criterias" 
            << std::endl;
    }

    bool bind(ObjFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* objFunc)
    {
        _checkOrQuit();

        try 
		{
            _innerOpt->set_min_objective(Optimizer::_nloptObjFunWrapper, objFunc);
            return true;
        } 
        catch (const std::exception& e) 
        {
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Unable to bind objective function: "
                << e.what() 
                << std::endl;
            return false;
        }
    }

    bool bindIneq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<AssignSize(sizeEnum::StateIneqSize)> tol)
    {
        _checkOrQuit();

        try 
		{
            if (_outputBounds) 
			{
                _innerOpt->add_inequality_mconstraint(
                            Optimizer::_nloptIneqConFunWrapper,
                            conFunc,
                            std::vector<double>(
                                tol.data(),
                                tol.data() + tol.rows() * tol.cols()));
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "Adding state defined inequality constraints" 
                    << std::endl;
            } 
            else 
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                << "State inequality constraints skipped" 
                << std::endl;
            }
            return true;
        } 
        catch (const std::exception& e) 
        {
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Unable to bind constraints function\n"
                << e.what() 
                << '\n';
            return false;
        }
    }

    bool bindEq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<AssignSize(sizeEnum::StateEqSize)> tol)
    {
        _checkOrQuit();

        try 
		{
            _innerOpt->add_equality_mconstraint(
                        Optimizer::_nloptEqConFunWrapper,
                        conFunc,
                        std::vector<double>(
                            tol.data(),
                            tol.data() + tol.rows() * tol.cols()));
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Adding state defined equality constraints" 
                << std::endl;
            return true;
        } 
        catch (const std::exception& e) 
        {
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Unable to bind constraints function\n"
                << e.what() 
                << '\n';
            return false;
        }
    }

    bool bindUserIneq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<Tineq> tol)
    {
        _checkOrQuit();

        try 
		{
            _innerOpt->add_inequality_mconstraint(
                        Optimizer::_nloptUserIneqConFunWrapper,
                        conFunc,
                        std::vector<double>(
                            tol.data(),
                            tol.data() + tol.rows() * tol.cols()));
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Adding user inequality constraints" 
                << std::endl;
            return true;
        } 
        catch (const std::exception& e) 
        {
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Unable to bind constraints function\n"
                << e.what() 
                << '\n';
            return false;
        }
    }

    bool bindUserEq(
            ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>* conFunc,
            constraints_type type,
            const cvec<Teq> tol)
    {
        _checkOrQuit();

        try 
		{
            _innerOpt->add_equality_mconstraint(
                        Optimizer::_nloptUserEqConFunWrapper,
                        conFunc,
                        std::vector<double>(
                            tol.data(),
                            tol.data() + tol.rows() * tol.cols()));
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Adding user equality constraints" 
                << std::endl;
            return true;
        } 
        catch (const std::exception& e) 
        {
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Unable to bind constraints function\n"
                << e.what() 
                << '\n';
            return false;
        }
    }

    typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::Result run(
            const cvec<Tnx> x0,
            const cvec<Tnu> u0)
    {
        _checkOrQuit();

        typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::Result r;

        static std::vector<double> optX0;
        optX0.resize(GetSize(sizeEnum::DecVarsSize));

        int counter = 0;
        for (int i = 0; i < _dimensions.tph; i++)
        {
            for (int j = 0; j < _dimensions.tnx; j++)
            {
                optX0[counter++] = x0(j);
            }
        }

        static mat<Tph, Tnu> Umv;
        Umv.resize(_dimensions.tph, _dimensions.tnu);
        Umv.setZero();

        for (int i = 0; i < _dimensions.tph; i++)
        {
            for (int j = 0; j < _dimensions.tnu; j++)
            {
                Umv(i, j) = u0(j);
            }
        }

        static cvec<AssignSize(sizeEnum::InputPredictionSize)> UmvVectorized;
        UmvVectorized.resize(GetSize(sizeEnum::InputPredictionSize));

        int vec_counter = 0;
        for (int i = 0; i < _dimensions.tph; i++)
        {
            for (int j = 0; j < _dimensions.tnu; j++)
            {
                UmvVectorized(vec_counter++) = Umv(i, j);
            }
        }

        static cvec<AssignSize(sizeEnum::InputEqSize)> res;
        res = _mapping.Iu2z() * UmvVectorized;

        for (int j = 0; j < GetSize(sizeEnum::InputEqSize); j++)
        {
            optX0[counter++] = res(j);
        }

        optX0[GetSize(sizeEnum::DecVarsSize) - 1] = _currentSlack;

        try 
		{
            auto res = _innerOpt->optimize(optX0);
            
            static cvec<AssignSize(sizeEnum::DecVarsSize)> opt_vector;
            opt_vector = Eigen::Map<cvec<AssignSize(sizeEnum::DecVarsSize)>>(res.data(), res.size());

            r.cost = _innerOpt->last_optimum_value();
            r.retcode = _innerOpt->last_optimize_result();

            Logger::instance().log(Logger::log_type::INFO) 
                << "Optimization end after: "
                << _innerOpt->get_numevals()
                << " evaluation steps" 
                << std::endl;
            Logger::instance().log(Logger::log_type::INFO) 
                << "Optimization end with code: "
                << r.retcode 
                << std::endl;
            Logger::instance().log(Logger::log_type::INFO) 
                << "Optimization end with cost: "
                << r.cost 
                << std::endl;

            static mat<AssignSize(sizeEnum::TphPlusOne), Tnx> Xmat;
            Xmat.resize(GetSize(sizeEnum::TphPlusOne), _dimensions.tnx);

            static mat<AssignSize(sizeEnum::TphPlusOne), Tnu> Umat;
            Umat.resize(GetSize(sizeEnum::TphPlusOne), _dimensions.tnu);

			_mapping.unwrapVector(opt_vector, x0, Xmat, Umat, _currentSlack);

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Optimal predicted state vector\n"
                << Xmat 
                << std::endl;
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "Optimal predicted output vector\n"
                << Umat 
                << std::endl;
            r.cmd = Umat.row(0);
        } 
        catch (const std::exception& e) 
        {
            Logger::instance().log(Logger::log_type::INFO) 
                << "No optimal solution found: " 
                << e.what() 
                << std::endl;
            r.cmd = _last_r.cmd;
            r.retcode = -1;
        }

        _last_r = r;
        return r;
    }

private:
    static double _nloptObjFunWrapper(
		const std::vector<double>& x, 
		std::vector<double>& grad, 
		void* objFunc)
    {
        bool hasGradient = !grad.empty();
        static cvec<AssignSize(sizeEnum::DecVarsSize)> x_arr;
        x_arr = Eigen::Map<cvec<AssignSize(sizeEnum::DecVarsSize)>>((double*) x.data(), x.size());

        auto res = ((ObjFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)objFunc)->evaluate(x_arr, hasGradient);

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

    static void _nloptIneqConFunWrapper(
		unsigned int m, 
		double* result, 
		unsigned int n, 
		const double* x, 
		double* grad, 
		void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        static cvec<AssignSize(sizeEnum::DecVarsSize)> x_arr;
        x_arr = Eigen::Map<cvec<AssignSize(sizeEnum::DecVarsSize)>>((double*) x, n);

        auto res = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc)->evaluateIneq(x_arr, hasGradient);
        
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

    static void _nloptEqConFunWrapper(
		unsigned int m, 
		double* result, 
		unsigned int n, 
		const double* x,
		double* grad, 
		void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        static cvec<AssignSize(sizeEnum::DecVarsSize)> x_arr;
        x_arr = Eigen::Map<cvec<AssignSize(sizeEnum::DecVarsSize)>>((double*) x, n);

        auto res = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc)->evaluateEq(x_arr, hasGradient);

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

    static void _nloptUserIneqConFunWrapper(
		unsigned int m, 
		double* result, 
		unsigned int n, 
		const double* x, 
		double* grad, 
		void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        static cvec<AssignSize(sizeEnum::DecVarsSize)> x_arr;
        x_arr = Eigen::Map<cvec<AssignSize(sizeEnum::DecVarsSize)>>((double*) x, n);

        auto res = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc)->evaluateUserIneq(x_arr, hasGradient);

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

    static void _nloptUserEqConFunWrapper(
		unsigned int m, 
		double* result, 
		unsigned int n, 
		const double* x, 
		double* grad, 
		void* conFunc)
    {
        bool hasGradient = (grad != NULL);
        static cvec<AssignSize(sizeEnum::DecVarsSize)> x_arr;
        x_arr = Eigen::Map<cvec<AssignSize(sizeEnum::DecVarsSize)>>((double*) x, n);

        auto res = static_cast<ConFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*>(conFunc)->evaluateUserEq(x_arr, hasGradient);

        //_mat2mem(result, res.value);
        //if (hasGradient)
        //{
        //    // The gradient should be transposed since the difference between matlab and nlopt
        //    _mat2mem(grad, res.grad.transpose());
        //}

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

    //template <typename Derived>
    //static inline void _mat2mem(double* dst, Eigen::MatrixBase<Derived> &src)
    //{
    //    std::memcpy(
    //        dst,
    //        src.data(),
    //        src.rows() * src.cols() * sizeof(double));
    //}

    nlopt::opt* _innerOpt;
    typename Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::Result _last_r;
    double _currentSlack;
    bool _hard;
    bool _outputBounds;

    Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> _mapping;
};
} // namespace mpc
