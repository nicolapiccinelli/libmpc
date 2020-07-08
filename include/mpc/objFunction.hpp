#pragma once

#include <mpc/baseFunction.hpp>

namespace mpc {
    template<
        int Tnx,
        int Tnu,
        int Tny,
        int Tph,
        int Tch,
        int Tineq,
        int Teq>
class ObjFunction :
        public BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>
{
    using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_mapping;
    using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_x0;
    using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_Xmat;
    using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_Umat;
    using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_e;
    using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_niteration;

    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_initialize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_checkOrQuit;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::GetSize;

    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_dimensions;

public:
    struct Cost
    {
        double value;
        cvec<AssignSize(sizeEnum::DecVarsSize)> grad;
    };

    ObjFunction() = default;

    ~ObjFunction() = default;

    void initialize(
        int tnx, int tnu, int tny,
        int tph, int tch,
        int tineq, int teq)
    {
        _initialize(tnx, tnu, tny, tph, tch, tineq, teq);

        _x0.resize(_dimensions.tnx);
        _Xmat.resize(GetSize(sizeEnum::TphPlusOne), _dimensions.tnx);
        _Umat.resize(GetSize(sizeEnum::TphPlusOne), _dimensions.tnu);
        _Jx.resize(_dimensions.tnx, _dimensions.tph);
        _Jmv.resize(_dimensions.tnu, _dimensions.tph);

        _Je = 0;
    }

    bool setUserFunction(
            const typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::ObjFunHandle handle)
    {
        _checkOrQuit();
        return _fuser = handle, true;
    }

    Cost evaluate(
            cvec<AssignSize(sizeEnum::DecVarsSize)> x,
            bool hasGradient)
    {
        _checkOrQuit();

        Cost c;
        c.grad.resize(GetSize(sizeEnum::DecVarsSize));

        _mapping.unwrapVector(x, _x0, _Xmat, _Umat, _e);
        c.value = _fuser(_Xmat, _Umat, _e);

        if (hasGradient)
        {
            _computeJacobian(_Xmat, _Umat, c.value, _e);

            int counter = 0;

            //#pragma omp parallel for
            for (int j = 0; j < (int)_Jx.cols(); j++)
            {
                for (int i = 0; i < (int)_Jx.rows(); i++)
                {
                    c.grad(counter++) = _Jx(i, j);
                }
            }

            static cvec<AssignSize(sizeEnum::InputPredictionSize)> JmvVectorized;
            JmvVectorized.resize(GetSize(sizeEnum::InputPredictionSize));

            int vec_counter = 0;
            //#pragma omp parallel for
            for (int j = 0; j < (int)_Jmv.cols(); j++)
            {
                for (int i = 0; i < (int)_Jmv.rows(); i++)
                {
                    JmvVectorized(vec_counter++) = _Jmv(i, j);
                }
            }

            static cvec<AssignSize(sizeEnum::InputEqSize)> res;
            res = _mapping.Iz2u().transpose() * JmvVectorized;
            //#pragma omp parallel for
            for (int j = 0; j < _dimensions.tch * _dimensions.tnu; j++)
            {
                c.grad(counter++) = res(j);
            }

            c.grad(GetSize(sizeEnum::DecVarsSize) - 1) = _Je;
        }

        // TODO support scaling

        Logger::instance().log(Logger::log_type::DEBUG) 
            << "(" 
            << _niteration 
            << ") Objective function value: \n"
            << std::setprecision(10) 
            << c.value 
            << std::endl;
        if (!hasGradient) 
        {
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "(" 
                << _niteration 
                << ") Gradient not currectly used"
                << std::endl;
        } 
        else 
        {
            Logger::instance().log(Logger::log_type::DEBUG) 
                << "(" 
                << _niteration 
                << ") Objective function gradient: \n"
                << std::setprecision(10) 
                << c.grad 
                << std::endl;
        }

        // debug information
        _niteration++;

        return c;
    }

private:
    void _computeJacobian(
            mat<AssignSize(sizeEnum::TphPlusOne), Tnx> x0,
            mat<AssignSize(sizeEnum::TphPlusOne), Tnu> u0,
            double f0,
            double e0)
    {
        double dv = 1e-6;

        _Jx.setZero();
        // TODO support measured disturbaces
        _Jmv.setZero();

        static mat<AssignSize(sizeEnum::TphPlusOne), Tnx> Xa;
        Xa = x0.cwiseAbs();

        //#pragma omp parallel for
        for (int i = 0; i < Xa.rows(); i++)
        {
            for (int j = 0; j < Xa.cols(); j++)
            {
                Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
            }
        }

        //#pragma omp parallel for
        for (int i = 0; i < _dimensions.tph; i++)
        {
            for (int j = 0; j < _dimensions.tnx; j++)
            {
                int ix = i + 1;
                double dx = dv * Xa(j, 0);
                x0(ix, j) = x0(ix, j) + dx;
                double f = _fuser(x0, u0, e0);
                x0(ix, j) = x0(ix, j) - dx;
                double df = (f - f0) / dx;
                _Jx(j, i) = df;
            }
        }

        static mat<AssignSize(sizeEnum::TphPlusOne), Tnu> Ua;
        Ua = u0.cwiseAbs();

        //#pragma omp parallel for
        for (int i = 0; i < Ua.rows(); i++)
        {
            for (int j = 0; j < Ua.cols(); j++)
            {
                Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
            }
        }

        //#pragma omp parallel for
        for (int i = 0; i < _dimensions.tph - 1; i++)
        {
            // TODO support measured disturbaces
            for (int j = 0; j < _dimensions.tnu; j++)
            {
                int k = j;
                double du = dv * Ua(k, 0);
                u0(i, k) = u0(i, k) + du;
                double f = _fuser(x0, u0, e0);
                u0(i, k) = u0(i, k) - du;
                double df = (f - f0) / du;
                _Jmv(j, i) = df;
            }
        }

        // TODO support measured disturbaces
        //#pragma omp parallel for
        for (int j = 0; j < _dimensions.tnu; j++)
        {
            int k = j;
            double du = dv * Ua(k, 0);
            u0(_dimensions.tph - 1, k) = u0(_dimensions.tph - 1, k) + du;
            u0(_dimensions.tph, k) = u0(_dimensions.tph, k) + du;
            double f = _fuser(x0, u0, e0);
            u0(_dimensions.tph - 1, k) = u0(_dimensions.tph - 1, k) - du;
            u0(_dimensions.tph, k) = u0(_dimensions.tph, k) - du;
            double df = (f - f0) / du;
            _Jmv(j, _dimensions.tph - 1) = df;
        }

        double ea = fmax(1e-6, abs(e0));
        double de = ea * dv;
        double f1 = _fuser(x0, u0, e0 + de);
        double f2 = _fuser(x0, u0, e0 - de);
        _Je = (f1 - f2) / (2 * de);
    }

    typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::ObjFunHandle _fuser = nullptr;

    mat<Tnx, Tph> _Jx;
    mat<Tnu, Tph> _Jmv;

    double _Je;
};
} // namespace mpc
