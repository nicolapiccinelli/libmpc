#pragma once

#include <mpc/Common.hpp>

namespace mpc {
template <
    int Tnx, int Tnu, int Tndu, int Tny,
    int Tph, int Tch, int Tineq, int Teq>
class IOptimizer : public Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq> {
public:
    virtual ~IOptimizer() {}    
    virtual void onInit() = 0;
    virtual void setParameters(const Parameters param) = 0;
    virtual Result<Tnu> run(const cvec<Tnx>& x0, const cvec<Tnu>& u0) = 0;
};
}