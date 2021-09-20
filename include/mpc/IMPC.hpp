#pragma once

#include <mpc/Common.hpp>
#include <mpc/IOptimizer.hpp>

#include <chrono>

namespace mpc {
/**
 * @brief Abstract class defining the shared API between linear
 * and non-linear MPC
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
class IMPC : public Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq> {

public:
    /**
     * @brief Set the discretization time step to use for numerical integration
     * 
     * @return true 
     * @return false 
     */
    virtual bool setContinuosTimeModel(const double) = 0;
    /**
     * @brief Set the scaling factor for the control input. This can be used to normalize
     * the control input with respect to the different measurment units
     */
    virtual void setInputScale(const cvec<Tnu>) = 0;
    /**
     * @brief Set the scaling factor for the dynamical system's states variables.
     * This can be used to normalize the dynamical system's states variables 
     * with respect to the different measurment units
     */
    virtual void setStateScale(const cvec<Tnx>) = 0;
    /**
     * @brief Set the solver specific parameters
     */
    virtual void setOptimizerParameters(const Parameters&) = 0;

    /**
     * @brief Implements the initilization hook to provide shared initilization logic
     * and forwards the hook through the setup hook for linear and non-linear interface
     * specific initilization
     */
    void onInit()
    {
        onSetup();

        result.cmd.resize(dim.nu.num());
        result.cmd.setZero();
    };

    /**
     * @brief Set the logger level
     * 
     * @param l logger level desired
     * @return true 
     * @return false 
     */
    bool setLoggerLevel(Logger::log_level l)
    {
        Logger::instance().setLevel(l);
        return true;
    }

    /**
     * @brief Set the prefix used on any log message
     * 
     * @param prefix prefix desired
     * @return true 
     * @return false 
     */
    bool setLoggerPrefix(std::string prefix)
    {
        Logger::instance().setPrefix(prefix);
        return true;
    }

    /**
     * @brief Compute the optimal control action
     * 
     * @param x0 system's variables initial condition
     * @param lastU last optimal control action
     * @return Result<Tnu> optimization result
     */
    Result<Tnu> step(const cvec<Tnx> x0, const cvec<Tnu> lastU)
    {
        checkOrQuit();

        onModelUpdate(x0);

        Logger::instance().log(Logger::log_type::INFO)
            << "Optimization step"
            << std::endl;

        auto start = std::chrono::steady_clock::now();
        result = optPtr->run(x0, lastU);
        auto stop = std::chrono::steady_clock::now();

        Logger::instance().log(Logger::log_type::INFO)
            << "Optimization step duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
            << " (ms)"
            << std::endl;
        return result;
    }

    /**
     * @brief Get the last optimal control action
     * 
     * @return Result<Tnu> last optimal control action
     */
    Result<Tnu> getLastResult()
    {
        checkOrQuit();
        return result;
    }

protected:
    /**
     * @brief Initilization hook for the linear and non-linear interfaces
     */
    virtual void onSetup() = 0;
    /**
     * @brief Dynamical system initial condition update hook
     */
    virtual void onModelUpdate(const cvec<Tnx>) = 0;

    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>::checkOrQuit;
    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>::dim;

    IOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>* optPtr;
    Result<Tnu> result;
};
}