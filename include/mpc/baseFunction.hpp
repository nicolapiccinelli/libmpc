#pragma once

#include <mpc/common.hpp>
#include <mpc/mapping.hpp>
#include <mpc/types.hpp>

namespace mpc {

template<
    int Tnx,
    int Tnu,
    int Tny,
    int Tph,
    int Tch,
    int Tineq,
    int Teq>
class BaseFunction :
    public Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>
{
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::GetSize;
    using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_dimensions;

public:
    BaseFunction() : Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>()
    {
        _e = 0;
        _ts = 0;
        _niteration = 0;

        _ctime = false;
    }

    bool setContinuos(bool isContinuous, double Ts = 0)
    {
        _ts = Ts;
        _ctime = isContinuous;
        return true;
    }

    void setMapping(Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>& m)
    {
        _mapping = m;
    }

    void setCurrentState(const cvec<Tnx> currState)
    {
        _x0 = currState;
        _niteration = 1;
    }

    // debug information
    int _niteration;

protected:
    Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> _mapping;

    cvec<Tnx> _x0;
    mat<AssignSize(sizeEnum::TphPlusOne), Tnx> _Xmat;
    mat<AssignSize(sizeEnum::TphPlusOne), Tnu> _Umat;

    double _e;

    double _ts;
    bool _ctime;
};

} // namespace mpc
