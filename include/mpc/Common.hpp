#pragma once

#include <mpc/Dim.hpp>
#include <mpc/Types.hpp>

namespace mpc {

template <
    int Tnx, int Tnu, int Tndu, int Tny,
    int Tph, int Tch, int Tineq, int Teq>
class Common {
public:
    Common()
    {
        isInitialized = false;
    }

    virtual void onInit() = 0;

    void initialize(
        int nx = Tnx, int nu = Tnu, int ndu = Tndu, int ny = Tny,
        int ph = Tph, int ch = Tch, int ineq = Tineq, int eq = Teq)
    {
        assert(nx >= 0 && nu >= 0 && ndu >= 0 && ny >= 0 && ph > 0 && ch > 0 && ineq >= 0 && eq >= 0);
        dim.set(nx, nu, ndu, ny, ph, ch, ineq, eq);
        isInitialized = true;

        onInit();
    }

    struct MPCDims {
        Dim<Tnx> nx;
        Dim<Tnu> nu;
        Dim<Tndu> ndu;
        Dim<Tny> ny;
        Dim<Tph> ph;
        Dim<Tch> ch;
        Dim<Tineq> ineq;
        Dim<Teq> eq;

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
