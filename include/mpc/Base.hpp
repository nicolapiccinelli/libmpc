#pragma once

#include <mpc/Common.hpp>
#include <mpc/Mapping.hpp>
#include <mpc/Types.hpp>

namespace mpc {

/**
 * @brief Abstract base class for the linear and non-linear mpc
 * interfaces
 * 
 * @tparam Tnx dimension of the state space
 * @tparam Tnu dimension of the input space
 * @tparam Tndu dimension of the measured disturbance space
 * @tparam Tny dimension of the output space
 * @tparam Tph length of the prediction horizon
 * @tparam Tch length of the control horizon
 * @tparam Tineq number of the user inequality constraints
 * @tparam Teq number of the user equality constraints
 */
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

    /**
     * @brief Initialization hook override used to perform mpc interfaces
     * initialization procedure. Performing initialization in this
     * method ensures the correct problem dimensions assigment has been
     * already performed
     */
    void onInit() = 0;

    /**
     * @brief Set the mapping object reference
     * 
     * @param m the mapping object
     */
    void setMapping(Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>& m)
    {
        mapping = m;
    }

    /**
     * @brief Set the current state of the optimizer
     * 
     * @param currState 
     */
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
