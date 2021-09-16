#pragma once

#include <mpc/Types.hpp>
#include <unsupported/Eigen/MatrixFunctions>

namespace mpc {
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

template <typename _Matrix_Type_>
_Matrix_Type_ pinv(_Matrix_Type_ A)
{
    Eigen::JacobiSVD<_Matrix_Type_> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV); //M=USV*
    double pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = std::min(row, col);
    _Matrix_Type_ X = _Matrix_Type_::Zero(col, row);
    _Matrix_Type_ singularValues_inv = svd.singularValues(); //singular value
    _Matrix_Type_ singularValues_inv_mat = _Matrix_Type_::Zero(col, row);
    for (long i = 0; i < k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else
            singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X = (svd.matrixV()) * (singularValues_inv_mat) * (svd.matrixU().transpose()); //X=VS+U*

    return X;
}

}