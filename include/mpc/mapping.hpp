#pragma once

#include <mpc/common.hpp>

namespace mpc {

template<
    int Tnx, int Tnu, int Tny,
    int Tph, int Tch,
    int Tineq, int Teq>
class Mapping :
    public Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>
{
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_initialize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_checkOrQuit;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::GetSize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_dimensions;

public:
    Mapping() : Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>(){}

    void initialize(
        int tnx = Tnx, int tnu = Tnu, int tny = Tny,
        int tph = Tph, int tch = Tch,
        int tineq = Tineq, int teq = Teq) 
    {
        _initialize(tnx, tnu, tny, tph, tch, tineq, teq);

        _Iz2u.resize(GetSize(sizeEnum::InputPredictionSize), GetSize(sizeEnum::InputEqSize));
        _Iu2z.resize(GetSize(sizeEnum::InputEqSize), GetSize(sizeEnum::InputPredictionSize));
        _Sz2u.resize(_dimensions.tnu, _dimensions.tnu);
        _Su2z.resize(_dimensions.tnu, _dimensions.tnu);

        _computeMapping();
    }

    mat<AssignSize(sizeEnum::InputPredictionSize), AssignSize(sizeEnum::InputEqSize)> Iz2u()
    {
        _checkOrQuit();
        return _Iz2u;
    }

    mat<AssignSize(sizeEnum::InputEqSize), AssignSize(sizeEnum::InputPredictionSize)> Iu2z()
    {
        _checkOrQuit();
        return _Iu2z;
    }

    mat<Tnu, Tnu> Sz2u()
    {
        _checkOrQuit();
        return _Sz2u;
    }

    mat<Tnu, Tnu> Su2z()
    {
        _checkOrQuit();
        return _Su2z;
    }

    void unwrapVector(
        const cvec<AssignSize(sizeEnum::DecVarsSize)> x,
        cvec<Tnx> x0, 
        mat<AssignSize(sizeEnum::TphPlusOne), Tnx>& Xmat,
        mat<AssignSize(sizeEnum::TphPlusOne), Tnu>& Umat,
        double& slack)
    {
        _checkOrQuit();

        static cvec<AssignSize(sizeEnum::InputEqSize)> u_vec;
        u_vec = x.middleRows(GetSize(sizeEnum::StateEqSize), GetSize(sizeEnum::InputEqSize));

        static mat<AssignSize(sizeEnum::TphPlusOne), Tnu> Umv;
        Umv.resize(GetSize(sizeEnum::TphPlusOne), _dimensions.tnu);

        static cvec<AssignSize(sizeEnum::InputPredictionSize)> tmp_mult;
        tmp_mult = Iz2u() * u_vec;
        static mat<Tnu, Tph> tmp_mapped;
        tmp_mapped = Eigen::Map<mat<Tnu, Tph>>(tmp_mult.data(), _dimensions.tnu, _dimensions.tph);

        Umv.setZero();
        Umv.middleRows(0, _dimensions.tph) = tmp_mapped.transpose();
        Umv.row(_dimensions.tph) = Umv.row(_dimensions.tph - 1);

        Xmat.setZero();
        Xmat.row(0) = x0.transpose();
        for (int i = 1; i < _dimensions.tph + 1; i++)
        {
            Xmat.row(i) = x.middleRows(((i - 1) * _dimensions.tnx), _dimensions.tnx).transpose();
        }
        // TODO add rows scaling

        // TODO add disturbaces manipulated vars
        Umat.setZero();
        Umat.block(0, 0, _dimensions.tph + 1, _dimensions.tnu) = Umv;

        slack = x(x.size() - 1);
    }

private:
    void _computeMapping()
    {
        static cvec<Tch> m;
        m.resize(_dimensions.tch);

        for (int i = 0; i < _dimensions.tch; i++)
        {
            m(i) = 1;
        }
        m(_dimensions.tch - 1) = _dimensions.tph - _dimensions.tch + 1;

        _Iz2u.setZero();
        _Iu2z = _Iz2u.transpose();

        _Sz2u.setZero();
        _Su2z.setZero();
        for (int i = 0; i < _Sz2u.rows(); ++i)
        {
            // TODO add scaling factor
            double scale = 1.0;
            _Sz2u(i, i) = scale;
            _Su2z(i, i) = 1.0 / scale;
        }

        // TODO implement linear interpolation
        int ix = 0;
        int jx = 0;
        for (int i = 0; i < _dimensions.tch; i++)
        {
            _Iu2z.block(ix, jx, _dimensions.tnu, _dimensions.tnu) = _Su2z;
            for (int j = 0; j < m[i]; j++)
            {
                _Iz2u.block(jx, ix, _dimensions.tnu, _dimensions.tnu) = _Sz2u;
                jx += _dimensions.tnu;
            }
            ix += _dimensions.tnu;
        }
    }

    mat<AssignSize(sizeEnum::InputPredictionSize), AssignSize(sizeEnum::InputEqSize)> _Iz2u;
    mat<AssignSize(sizeEnum::InputEqSize), AssignSize(sizeEnum::InputPredictionSize)> _Iu2z;
    mat<Tnu, Tnu> _Sz2u;
    mat<Tnu, Tnu> _Su2z;
};
} // namespace mpc
