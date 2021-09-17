#pragma once

#include <mpc/Dim.hpp>
#include <mpc/Types.hpp>

namespace mpc {

/**
 * @brief Abstract base class for the classes which need access
 * to the problem dimensions and to the function handlers types
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
class Common {
public:
    Common()
    {
        isInitialized = false;
    }

    /**
     * @brief Initialization hook used to perform sub-classes
     * initialization procedure. Performing initialization in this
     * method ensures the correct problem dimensions assigment has been
     * already performed
     */
    virtual void onInit() = 0;

    /**
     * @brief Initialize the dimensions of the optimization problem
     * and then invokes the onInit method to perform extra initialization.
     * In case of static allocation the dimensions are inferred from the
     * template class parameters
     * 
     * @param nx dimension of the state space
     * @param nu dimension of the input space
     * @param ndu dimension of the measured disturbance space
     * @param ny dimension of the output space
     * @param ph length of the prediction horizon
     * @param ch length of the control horizon
     * @param ineq number of the user inequality constraints
     * @param eq number of the user equality constraints
     */
    void initialize(
        int nx = Tnx, int nu = Tnu, int ndu = Tndu, int ny = Tny,
        int ph = Tph, int ch = Tch, int ineq = Tineq, int eq = Teq)
    {
        assert(nx >= 0 && nu >= 0 && ndu >= 0 && ny >= 0 && ph > 0 && ch > 0 && ineq >= 0 && eq >= 0);
        dim.set(nx, nu, ndu, ny, ph, ch, ineq, eq);
        isInitialized = true;

        onInit();
    }

    /**
     * @brief The problem dimensions structure containing
     * the instances of each dimension for static or dynamic
     * access
     */
    struct MPCDims {
        /**
         * @brief Dimension of the state space
         */
        Dim<Tnx> nx;
        /**
         * @brief Dimension of the input space
         */
        Dim<Tnu> nu;
        /**
         * @brief Dimension of the measured disturbance space
         */
        Dim<Tndu> ndu;
        /**
         * @brief Dimension of the output space
         */
        Dim<Tny> ny;
        /**
         * @brief Dimension of the prediction horizon
         */
        Dim<Tph> ph;
        /**
         * @brief Dimension of the control horizon
         */
        Dim<Tch> ch;
        /**
         * @brief Number of the user inequality constraints
         */
        Dim<Tineq> ineq;
        /**
         * @brief Number of the user equality constraints
         */
        Dim<Teq> eq;

        /**
         * @brief Set the dynamic dimensions
         * 
         * @param nx Dimension of the state space
         * @param nu Dimension of the input space
         * @param ndu Dimension of the measured disturbance space
         * @param ny Dimension of the output space
         * @param ph Length of the prediction horizon
         * @param ch Length of the control horizon
         * @param ineq Number of the user inequality constraints
         * @param eq Number of the user equality constraints
         */
        void set(
            size_t nx, size_t nu, size_t ndu, size_t ny,
            size_t ph, size_t ch, size_t ineq, size_t eq)
        {
            this->nx.setDynDim(nx);
            this->nu.setDynDim(nu);
            this->ndu.setDynDim(ndu);
            this->ny.setDynDim(ny);
            this->ph.setDynDim(ph);
            this->ch.setDynDim(ch);
            this->ineq.setDynDim(ineq);
            this->eq.setDynDim(eq);
        }
    };

    inline static MPCDims dim;

protected:

    /**
     * @brief Check if the object has been correctly initialized. In case
     * the initialization has not been performed yet, the library exits
     * causing a crash
     */
    inline void checkOrQuit()
    {
        if (!isInitialized) {
            Logger::instance().log(Logger::log_type::ERROR) << RED << "MPC library is not initialized, quitting..." << RESET << std::endl;
            exit(-1);
        }
    }

    using ObjFunHandle = std::function<double(
        mat<dim.ph + Dim<1>(), dim.nx>,
        mat<dim.ph + Dim<1>(), dim.nu>,
        double)>;

    using IConFunHandle = std::function<void(
        cvec<dim.ineq>&,
        mat<dim.ph + Dim<1>(), dim.nx>,
        mat<dim.ph + Dim<1>(), dim.ny>,
        mat<dim.ph + Dim<1>(), dim.nu>,
        double)>;

    using EConFunHandle = std::function<void(
        cvec<dim.eq>&,
        mat<dim.ph + Dim<1>(), dim.nx>,
        mat<dim.ph + Dim<1>(), dim.nu>)>;

    using StateFunHandle = std::function<void(
        cvec<dim.nx>&,
        cvec<dim.nx>,
        cvec<dim.nu>)>;

    using OutFunHandle = std::function<void(
        mat<dim.ph + Dim<1>(), dim.ny>&,
        mat<dim.ph + Dim<1>(), dim.nx>,
        mat<dim.ph + Dim<1>(), dim.nu>)>;

private:
    bool isInitialized;
};

} // namespace mpc
