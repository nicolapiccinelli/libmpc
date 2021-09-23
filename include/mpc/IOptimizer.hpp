#pragma once

#include <mpc/IComponent.hpp>

namespace mpc {
/**
 * @brief Abstract class defining the optimizer interface for
 * linear and non-linear MPC
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
template <
    int Tnx, int Tnu, int Tndu, int Tny,
    int Tph, int Tch, int Tineq, int Teq>
class IOptimizer : public IComponent<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq> {
public:
    virtual ~IOptimizer() {}

    /**
     * @brief Initialization hook override used to perform the optimizer interfaces
     * initialization procedure. Performing initialization in this
     * method ensures the correct problem dimensions assigment has been
     * already performed
     */
    virtual void onInit() = 0;
    /**
     * @brief Abstract setter for the optimizer parameters
     * 
     * @param param desired parameters (refers to the optimizer specific for the meaning of the parameters)
     */
    virtual void setParameters(const Parameters& param) = 0;
    /**
     * @brief Abstract caller to perform the optimization
     * 
     * @param x0 system's variables initial condition
     * @param u0 control action initial condition for warm start
     * @return Result<Tnu> optimization result
     */
    virtual Result<Tnu> run(const cvec<Tnx>& x0, const cvec<Tnu>& u0) = 0;
};
}