#pragma once

#include <mpc/IMPC.hpp>

#include <mpc/Constraints.hpp>
#include <mpc/Objective.hpp>
#include <mpc/NLOptimizer.hpp>

namespace mpc {
/**
 * @brief Non-lnear MPC front-end class
 * 
 * @tparam Tnx dimension of the state space
 * @tparam Tnu dimension of the input space
 * @tparam Tny dimension of the output space
 * @tparam Tph length of the prediction horizon
 * @tparam Tch length of the control horizon
 * @tparam Tineq number of the user inequality constraints
 * @tparam Teq number of the user equality constraints
 */
template <
    int Tnx = Eigen::Dynamic, int Tnu = Eigen::Dynamic, int Tny = Eigen::Dynamic,
    int Tph = Eigen::Dynamic, int Tch = Eigen::Dynamic,
    int Tineq = Eigen::Dynamic, int Teq = Eigen::Dynamic>
class NLMPC : public IMPC<Tnx, Tnu,0, Tny, Tph, Tch, Tineq, Teq> 
{
    
private:
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::optPtr;
    using IDimensionable<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::setDimension;
    using IDimensionable<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::dim;

public:
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::step;
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::setLoggerLevel;
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::setLoggerPrefix;
    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::getLastResult;

    NLMPC()
    {
        setDimension();
    }

    NLMPC(
        const int& nx, const int& nu, const int& ny, 
        const int& ph, const int& ch, const int& ineq, const int& eq)
    {
        setDimension(nx, nu, 0, ny, ph, ch, ineq, eq);
    }

    ~NLMPC()
    {
        delete optPtr;
    }

    /**
     * @brief Set the discretization time step to use for numerical integration
     * 
     * @param ts sample time in seconds
     * @return true 
     * @return false 
     */
    bool setContinuosTimeModel(const double ts)
    {
        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting sampling time to: "
            << ts
            << " sec(s)"
            << std::endl;

        auto res = conF.setContinuos(true, ts);
        return res;
    }

    /**
     * @brief  Set the solver specific parameters
     * 
     * @param param desired parameters (the structure must be of type NLParameters)
     */
    void setOptimizerParameters(const Parameters& param)
    {
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setParameters(param);
    }

    /**
     * 
     * @brief Set the scaling factor for the control input
     * 
     * @param scaling scaling vector
     */
    void setInputScale(const cvec<Tnu> scaling)
    {
        mapping.setInputScaling(scaling);

        objF.setMapping(mapping);
        conF.setMapping(mapping);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setMapping(mapping);
    }

    /**
     * 
     * @brief Set the scaling factor for the dynamical system's states variables
     * 
     * @param scaling scaling vector
     */
    void setStateScale(const cvec<Tnx> scaling)
    {
        mapping.setStateScaling(scaling);

        objF.setMapping(mapping);
        conF.setMapping(mapping);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setMapping(mapping);
    }

    /**
     * @brief Set the handler to the function defining the objective function
     * 
     * @param handle function handler
     * @return true 
     * @return false 
     */
    bool setObjectiveFunction(const typename IDimensionable<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::ObjFunHandle handle)
    {
        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting objective function handle"
            << std::endl;

        auto res = objF.setObjective(handle);

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Binding objective function handle"
            << std::endl;

        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bind(&objF);
        return res;
    }

    /**
     * @brief Set the handler to the function defining the state space update function.
     * Based on the type of system (continuos or discrete) you should provide the appropriate
     * vector field differential equations or the finite differences update model
     * 
     * @param handle function handler
     * @param eq_tol equality constraints tolerances (default 1e-10)
     * @return true 
     * @return false 
     */
    bool setStateSpaceFunction(const typename IDimensionable<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::StateFunHandle handle,
        const float eq_tol = 1e-10)
    {
        static cvec<Dim<2>() * dim.ph * dim.ny> ineq_tol_vec;
        ineq_tol_vec.resize((2 * dim.ph.num() * dim.ny.num()));
        ineq_tol_vec.setOnes();

        static cvec<(dim.ph * dim.nx)> eq_tol_vec;
        eq_tol_vec.resize((dim.ph.num() * dim.nx.num()));
        eq_tol_vec.setOnes();

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting state space function handle"
            << std::endl;

        bool res = conF.setStateModel(handle);

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Binding state space constraints"
            << std::endl;

        res = res & ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bindEq(&conF, constraints_type::EQ, eq_tol_vec * eq_tol);

        return res;
    }

    /**
     * Set the handler to the function defining the output function
     * 
     * @param handle function handler
     * @return true 
     * @return false 
     */
    bool setOutputFunction(const typename IDimensionable<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::OutFunHandle handle)
    {        
        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting output function handle"
            << std::endl;

        return conF.setOutputModel(handle);
    }

    /**
     * @brief Set the handler to the function defining the user inequality constraints
     * 
     * @param handle function handler
     * @param tol inequality constraints tolerances (default 1e-10)
     * @return true 
     * @return false 
     */
    bool setIneqConFunction(
        const typename IDimensionable<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle handle, const float tol = 1e-10)
    {
        cvec<Tineq> tol_vec;
        tol_vec = cvec<Tineq>::Ones(dim.ineq.num());

        auto res = conF.setIneqConstraints(handle);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bindUserIneq(&conF, constraints_type::UINEQ, tol_vec * tol);
        return res;
    }

    /**
     * @brief Set the handler to the function defining the user equality constraints
     * 
     * @param handle function handler
     * @param tol equality constraints tolerances (default 1e-10)
     * @return true 
     * @return false 
     */
    bool setEqConFunction(
        const typename IDimensionable<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle handle, const float tol = 1e-10)
    {
        cvec<Teq> tol_vec;
        tol_vec = cvec<Teq>::Ones(dim.eq);

        auto res = conF.setEqConstraints(handle);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->bindUserEq(&conF, constraints_type::UEQ, tol_vec * tol);
        return res;
    }

protected:
    /**
     * @brief Initilization hook for the linear interface
     */
    void onSetup()
    {
        conF.initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        mapping.initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        objF.initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        optPtr = new NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>();
        optPtr->initialize(
            dim.nx.num(), dim.nu.num(), 0, dim.ny.num(),
            dim.ph.num(), dim.ch.num(), dim.ineq.num(),
            dim.eq.num());

        objF.setMapping(mapping);
        conF.setMapping(mapping);
        ((NLOptimizer<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>*)optPtr)->setMapping(mapping);

        Logger::instance().log(Logger::log_type::INFO)
            << "Mapping assignment done"
            << std::endl;
    }

    /**
     * @brief Dynamical system initial condition update hook
     */
    void onModelUpdate(const cvec<Tnx> x0)
    {
        objF.setCurrentState(x0);
        conF.setCurrentState(x0);
    }

private:
    Objective<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> objF;
    Constraints<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> conF;
    Mapping<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> mapping;

    using IMPC<Tnx, Tnu, 0, Tny, Tph, Tch, Tineq, Teq>::result;
};

} // namespace mpc
