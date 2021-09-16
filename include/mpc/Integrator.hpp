#pragma once

#include <mpc/Types.hpp>

namespace mpc{

// N4 runge-kutta
template <unsigned int N>
class RK4 {
public:
    RK4(std::function<cvec<N>(double, const cvec<N>&)> f)
        : m_func(f)
    {
    }

    cvec<N> run(double t, const cvec<N>& in, double h, int integration_step = 1)
    {
        cvec<N> sol = in;
        cvec<N> k1, k2, k3, k4;

        for (size_t i = 0; i < integration_step; i++) {
            k1 = m_func(t, sol);
            k2 = m_func(t + h / 2.0, sol + (h / 2.0) * k1);
            k3 = m_func(t + h / 2.0, sol + (h / 2.0) * k2);
            k4 = m_func(t + h, sol + h * k3);

            sol += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        }

        return sol;
    }

private:
    std::function<cvec<N>(double, const cvec<N>&)> m_func;
};
}