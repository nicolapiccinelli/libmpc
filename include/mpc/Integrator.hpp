/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/Types.hpp>

namespace mpc{

/**
 * @brief Numerical integration using Runge-Kutta 4th order
 * 
 * @tparam N dimension of the state vector
 */
template <unsigned int N>
class RK4 {
public:
    /**
     * @brief Construct a new RK4 object
     * 
     * @param f function handle to the system's dynamics to be integrated
     */
    RK4(std::function<cvec<N>(double, const cvec<N>&)> f) : m_func(f)
    {
    }

    /**
     * @brief Compute the numerical integration of the sytem's dynamics provided
     * 
     * @param t initial integration time
     * @param in initial condition
     * @param h integration step
     * @param integration_step number of integration step to perform multistep integration
     * @return cvec<N> integration result
     */
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