#pragma once

#include <mpc/mapping.hpp>
#include <mpc/mpc.hpp>

namespace mpc {
template <std::size_t Tnx, std::size_t Tnu, std::size_t Tph, std::size_t Tch>
class BaseFunction {
public:
    bool setContinuos(bool isContinuos, double Ts = 0)
    {
        ts = Ts;
        ctime = isContinuos;
        return true;
    }

    void setMapping(Common<Tnx, Tnu, Tph, Tch>& m)
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
    Common<Tnx, Tnu, Tph, Tch> mapping;
    cvec<Tnx> x0;

    mat<Tph + 1, Tnx> Xmat;
    mat<Tph + 1, Tnu> Umat;
    double e;

    double ts;
    bool ctime;
};
} // namespace mpc