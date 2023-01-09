/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/Types.hpp>
#include <unsupported/Eigen/MatrixFunctions>

namespace mpc {

/**
 * @brief Discretization of the linear system vector field
 * 
 * @tparam nx dimension of the state space
 * @tparam nu dimension of the input space
 * @param A state update matrix
 * @param B input matrix
 * @param Ts sampling time in seconds
 * @param Ad discrete time state update matrix
 * @param Bd discrete time input matrix
 */
template <size_t nx, size_t nu>
void discretization(
    const mat<nx, nx>& A, const mat<nx, nu>& B, const double& Ts,
    mat<nx, nx>& Ad, mat<nx, nu>& Bd)
{
    // concatenating the state and input matrices
    mat<nx, nx + nu> AB;
    AB.block(0, 0, nx, nx) = A;
    AB.block(0, nx, nx, nu) = B;
    
    // setting the time step
    AB = AB * Ts;

    // building the square matrix
    mat<nx + nu, nx + nu> res;
    res.setZero();
    res.block(0, 0, nx, nx + nu) = AB;

    // computing the exponential matrix
    res = res.exp();

    // retriving the discrete time matrices
    Ad = res.block(0, 0, nx, nx);
    Bd = res.block(0, nx, nx, nu);
}

/**
 * @brief Discretization of the linear system vector field
 * with input disturbances
 * 
 * @tparam nx dimension of the state space
 * @tparam nu dimension of the input space
 * @param A state update matrix
 * @param B input matrix
 * @param Be disturbances input matrix
 * @param Ts sampling time in seconds
 * @param Ad discrete time state update matrix
 * @param Bd discrete time input matrix
 * @param Be discrete time disturbances input matrix
 */
template <size_t nx, size_t nu, size_t nud>
void discretization(
    const mat<nx, nx>& A, const mat<nx, nu>& B, const mat<nx, nud>& Be, const double& Ts,
    mat<nx, nx>& Ad, mat<nx, nu>& Bd, mat<nx, nud>& Bed)
{
    // concatenating the state and input matrices
    mat<nx, nx + nu + nud> ABBe;
    ABBe.block(0, 0, nx, nx) = A;
    ABBe.block(0, nx, nx, nu) = B;
    ABBe.block(0, nx + nu, nx, nud) = Be;

    // setting the time step
    ABBe = ABBe * Ts;

    // building the square matrix
    mat<nx + nu + nud, nx + nu + nud> res;
    res.setZero();
    res.block(0, 0, nx, nx + nu + nud) = ABBe;

    // computing the exponential matrix
    res = res.exp();

    // retriving the discrete time matrices
    Ad = res.block(0, 0, nx, nx);
    Bd = res.block(0, nx, nx, nu);
    Bed = res.block(0, nx + nu, nx, nud);
}

/**
 * @brief  Discretization of the linear system vector field
 * and the output map
 * 
 * @tparam nx dimension of the state space
 * @tparam nu dimension of the input space
 * @tparam ny dimension of the output space
 * @param A state update matrix
 * @param B input matrix
 * @param C output matrix
 * @param D feedforward input matrix
 * @param Ts sampling time in seconds
 * @param Ad discrete time state update matrix
 * @param Bd discrete time input matrix
 * @param Cd discrete time output matrix
 * @param Dd discrete time feedforward input matrix
 */
template <size_t nx, size_t nu, size_t ny>
void discretization(
    const mat<nx, nx>& A, const mat<nx, nu>& B,
    const mat<ny, nx>& C, const mat<ny, nu>& D, const double& Ts,
    mat<nx, nx>& Ad, mat<nx, nu>& Bd, mat<nx, nx>& Cd, mat<nx, nu>& Dd)
{
    // TODO this discretization works only with delay-free systems
    discretization(A, B, Ts, Ad, Bd);
    Cd = C;
    Dd = D;
}
}