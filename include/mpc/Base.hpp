#pragma once

#include <mpc/Common.hpp>
#include <mpc/Mapping.hpp>
#include <mpc/Types.hpp>

namespace mpc {

template <int Tnx, int Tnu, int Tndu, int Tny,
    int Tph, int Tch, int Tineq, int Teq>
class Base : public Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq> {

public:
    Base()
        : Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>()
    {
        e = 0;
        ts = 0;
        niteration = 0;
    }

    void onInit() = 0;

    void setMapping(Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>& m)
    {
        mapping = m;
    }

    void setCurrentState(const cvec<Tnx> currState)
    {
        x0 = currState;
        niteration = 1;
    }

    // debug information
    int niteration;

protected:
    Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> mapping;
    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>::dim;

    cvec<dim.nx> x0;
    mat<(dim.ph + Dim<1>()), dim.nx> Xmat;
    mat<(dim.ph + Dim<1>()), dim.nu> Umat;

    double e;
    double ts;
};

} // namespace mpc
