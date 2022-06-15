#pragma once

#include <mpc/IComponent.hpp>
#include <mpc/Utils.hpp>
#include <unsupported/Eigen/KroneckerProduct>

namespace mpc {
/**
 * @brief Linear MPC optimal problem builder
 * 
 * @tparam Tnx dimension of the state space
 * @tparam Tnu dimension of the input space
 * @tparam Tndu dimension of the measured disturbance space
 * @tparam Tny dimension of the output space
 * @tparam Tph length of the prediction horizon
 * @tparam Tch length of the control horizon
 */
template <int Tnx, int Tnu, int Tndu, int Tny, int Tph, int Tch>
class ProblemBuilder : public IComponent<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0> {
private:
    using IComponent<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::checkOrQuit;
    using IComponent<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::dim;

public:

    /**
     * @brief Linear optimal problem
     */
    class Problem {
    public:
        Problem() = default;
        Problem(const Problem &) = delete;
        Problem &operator=(const Problem &) = delete;

        /**
         * @brief Get the sparse matrices
         * 
         * @param Psparse objective function P matrix
         * @param Asparse constraints A matrix
         */
        void getSparse(smat& Psparse, smat& Asparse) const
        {
            // converting P matrix to sparse and
            // getting the upper triangular part of the matrix
            Psparse = P.sparseView();
            Psparse = Psparse.triangularView<Eigen::Upper>();

            // converting A matrix to sparse
            Asparse = A.sparseView();

            // let the matrix sparse
            Psparse.makeCompressed();
            Asparse.makeCompressed();
        }

        // objective_matrix is P
        mat<(((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (dim.ph * dim.nu)), (((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (dim.ph * dim.nu))> P;
        // objective_vector is q
        cvec<(((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (dim.ph * dim.nu))> q;
        // constraint_matrix is A
        mat<(((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (((dim.ph + Dim<1>()) * dim.ny) + (dim.ph * dim.nu)))), (((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (dim.ph * dim.nu))> A;
        // lower_bounds is l and upper_bounds is u
        cvec<(((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (((dim.ph + Dim<1>()) * dim.ny) + (dim.ph * dim.nu))))> l, u;
    };

    ProblemBuilder() = default;
    ~ProblemBuilder() = default;

    /**
     * @brief Initialization hook override used to perform the
     * initialization procedure. Performing initialization in this
     * method ensures the correct problem dimensions assigment has been
     * already performed.
     */
    void onInit()
    {
        ssA.resize(dim.nu.num() + dim.nx.num(), dim.nu.num() + dim.nx.num());
        ssB.resize(dim.nu.num() + dim.nx.num(), dim.nu.num());
        ssC.resize(dim.nu.num() + dim.ny.num(), dim.nu.num() + dim.nx.num());
        ssBv.resize(dim.nu.num() + dim.nx.num(), dim.ndu.num());
        ssDv.resize(dim.nu.num() + dim.ny.num(), dim.ndu.num());

        wOutput.resize(dim.ny.num(), (dim.ph.num() + 1));
        wU.resize(dim.nu.num(), (dim.ph.num() + 1));
        wDeltaU.resize(dim.nu.num(), dim.ph.num());

        minX.resize(dim.nx.num(), (dim.ph.num() + 1));
        maxX.resize(dim.nx.num(), (dim.ph.num() + 1));

        minY.resize(dim.ny.num(), (dim.ph.num() + 1));
        maxY.resize(dim.ny.num(), (dim.ph.num() + 1));

        minU.resize(dim.nu.num(), dim.ph.num());
        maxU.resize(dim.nu.num(), dim.ph.num());

        leq.resize(((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())));
        ueq.resize(((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())));

        lineq.resize((((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num()))));
        uineq.resize((((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num()))));

        ssA.setZero();
        ssB.setZero();
        ssC.setZero();
        ssBv.setZero();
        ssDv.setZero();

        wOutput.setZero();
        wU.setZero();
        wDeltaU.setZero();

        minX.setZero();
        maxX.setZero();
        minY.setZero();
        maxY.setZero();
        minU.setZero();
        maxU.setZero();

        leq.setZero();
        ueq.setZero();
        lineq.setZero();
        uineq.setZero();

        mpcProblem.P.resize(
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (dim.ph.num() * dim.nu.num())),
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (dim.ph.num() * dim.nu.num())));
        mpcProblem.q.resize(
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (dim.ph.num() * dim.nu.num())));
        mpcProblem.A.resize(
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num()))),
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (dim.ph.num() * dim.nu.num())));
        mpcProblem.l.resize(
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num())))));
        mpcProblem.u.resize(
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num())))));

        mpcProblem.P.setZero();
        mpcProblem.q.setZero();
        mpcProblem.A.setZero();
        mpcProblem.l.setZero();
        mpcProblem.u.setZero();
    }

    /**
     * @brief Set the state space model matrices
     * x(k+1) = A*x(k) + B*u(k) + Bd*d(k)
     * y(k) = C*x(k) + Dd*d(k)
     * @param A state update matrix
     * @param B input matrix
     * @param C output matrix
     * @return true 
     * @return false 
     */
    bool setStateModel(
        const mat<Tnx, Tnx>& A, const mat<Tnx, Tnu>& B,
        const mat<Tny, Tnx>& C)
    {
        checkOrQuit();

        // augmenting to system to store the command input of the current timestep
        ssA.block(0, 0, dim.nx.num(), dim.nx.num()) = A;
        ssA.block(0, dim.nx.num(), dim.nx.num(), dim.nu.num()) = B;
        ssA.block(dim.nx.num(), 0, dim.nu.num(), dim.nx.num()).setZero();
        ssA.block(dim.nx.num(), dim.nx.num(), dim.nu.num(), dim.nu.num()).setIdentity();

        ssB.block(0, 0, dim.nx.num(), dim.nu.num()) = B;
        ssB.block(dim.nx.num(), 0, dim.nu.num(), dim.nu.num()).setIdentity();

        // we put on the output also the command to allow its penalization
        ssC.block(0, 0, dim.ny.num(), dim.nx.num()) = C;
        ssC.block(dim.ny.num(), dim.nx.num(), dim.nu.num(), dim.nu.num()).setIdentity();

        return buildTITerms();
    }

    /**
     * @brief Set the disturbances matrices
     * x(k+1) = A*x(k) + B*u(k) + Bd*d(k)
     * y(k) = C*x(k) + Dd*d(k)
     * @param Bd state disturbance matrix 
     * @param Dd output disturbance matrix
     * @return true 
     * @return false 
     */
    bool setExogenuosInput(
        const mat<Tnx, Tndu>& Bd,
        const mat<Tny, Tndu>& Dd)
    {
        checkOrQuit();

        // the exogenous inputs goes only to states and outputs
        ssBv.setZero();
        ssBv.block(0, 0, dim.nx.num(), dim.ndu.num()) = Bd;

        ssDv.setZero();
        ssDv.block(0, 0, dim.ny.num(), dim.ndu.num()) = Dd;

        return buildTITerms();
    }

    /**
     * @brief Set the objective function weights
     * 
     * @param OWeight weights for the output vector
     * @param UWeight weights for the optimal control input vector
     * @param DeltaUWeight weight for the variation of the optimal control input vector
     * @return true 
     * @return false 
     */
    bool setObjective(
        const mat<Tny, (dim.ph + Dim<1>())>& OWeight,
        const mat<Tnu, (dim.ph + Dim<1>())>& UWeight,
        const mat<Tnu, Tph>& DeltaUWeight)
    {
        checkOrQuit();

        wOutput = OWeight;
        wU = UWeight;
        wDeltaU = DeltaUWeight;

        return buildTITerms();
    }

    /**
     * @brief Set the state, input and output box constraints
     * 
     * @param XMin minimum state vector
     * @param UMin minimum input vector
     * @param YMin minimum output vector
     * @param XMax maximum state vector
     * @param UMax maximum input vector
     * @param YMax maximum output vector
     * @return true 
     * @return false 
     */
    bool setConstraints(
        const mat<Tnx, Tph>& XMin, const mat<Tnu, Tph>& UMin, const mat<Tny, Tph>& YMin,
        const mat<Tnx, Tph>& XMax, const mat<Tnu, Tph>& UMax, const mat<Tny, Tph>& YMax)
    {
        checkOrQuit();

        minX.block(0, 1, dim.nx.num(), dim.ph.num()) = XMin;
        minX.col(0) = XMin.col(0);
        maxX.block(0, 1, dim.nx.num(), dim.ph.num()) = XMax;
        maxX.col(0) = XMax.col(0);

        minY.block(0, 1, dim.ny.num(), dim.ph.num()) = YMin;
        minY.col(0) = YMin.col(0);
        maxY.block(0, 1, dim.ny.num(), dim.ph.num()) = YMax;
        maxY.col(0) = YMax.col(0);

        minU = UMin;
        maxU = UMax;

        return buildTITerms();
    }

    /**
     * @brief Request the generation of a new MPC optimization problem
     * 
     * @param x0 initial condition of the system's dynamics vector
     * @param yRef output reference vector
     * @param uRef control input reference vector
     * @param deltaURef control input variation reference vector
     * @param uMeas external disturbances vector
     */
    const Problem& get(
        const cvec<Tnx>& x0,
        const cvec<Tnu>& /*u0*/,
        const cvec<Tny>& yRef,
        const cvec<Tnu>& uRef,
        const cvec<Tnu>& deltaURef,
        const cvec<Tndu>& uMeas)
    {
        // linear objective terms must be computed at each control loop since
        // it depends on the references and the refs can changes over time
        mat<(dim.ny + dim.nu), (dim.ny + dim.nu)> wExtendedState;
        wExtendedState.resize((dim.ny.num() + dim.nu.num()), (dim.ny.num() + dim.nu.num()));
        wExtendedState.setZero();

        cvec<(dim.ny + dim.nu)> eRef;
        eRef.resize(dim.ny.num() + dim.nu.num());
        eRef << yRef, uRef;

        mpcProblem.q.setZero();
        leq.setZero();

        for (size_t i = 0; i < dim.ph.num() + 1; i++) {
            wExtendedState.block(0, 0, dim.ny.num(), dim.ny.num()) = wOutput.col(i).asDiagonal();
            wExtendedState.block(
                dim.ny.num(), dim.ny.num(),
                dim.nu.num(), dim.nu.num())
                = wU.col(i).asDiagonal();

            mpcProblem.q.middleRows(
                i * (dim.nx.num() + dim.nu.num()), dim.nx.num() + dim.nu.num())
                = ssC.transpose() * wExtendedState * (-eRef + (ssDv * uMeas));

            // the command increments stop at the last prediction horizon step
            if (i < dim.ph.num()) {
                mpcProblem.q.middleRows(
                    ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (i * dim.nu.num()),
                    dim.nu.num())
                    = -(wDeltaU.col(i).asDiagonal() * deltaURef);
            }

            // the first entry of the state evolution is the initial condition of the states
            if (i > 0) {
                leq.middleRows(i * (dim.nx.num() + dim.nu.num()), dim.nx.num() + dim.nu.num()) = -ssBv * uMeas;
            }

            // let's add on the output part of the system
            // any contribution of the exogenous inputs on the output function
            lineq.middleRows(
                (i * dim.ny.num()) + ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
                dim.ny.num())
                -= ssDv.block(0, 0, dim.ny.num(), dim.ndu.num()) * uMeas;

            uineq.middleRows(
                (i * dim.ny.num()) + ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
                dim.ny.num())
                -= ssDv.block(0, 0, dim.ny.num(), dim.ndu.num()) * uMeas;
        }

        // state evolution depends on the initial condition and
        // on the exogeneous inputs so they are changes over time
        leq.middleRows(0, dim.nx.num()) = -x0;

        // set lower and upper bound equal in order to have equality constraints
        ueq = leq;

        // creation of bounds and here is the right place to take into account
        // of the measured exogenous inputs on the output (we are gonna treat them as offsets)
        mpcProblem.l.setZero();
        mpcProblem.u.setZero();

        mpcProblem.l.middleRows(
            0, (dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num()))
            = leq;

        mpcProblem.u.middleRows(
            0, (dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num()))
            = ueq;

        mpcProblem.l.middleRows(
            (dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num()),
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + ((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num()))
            = lineq;

        mpcProblem.u.middleRows(
            (dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num()),
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + ((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num()))
            = uineq;

        return mpcProblem;
    }

private:
    /**
     * @brief Build the time invariant optimal control problem terms
     * 
     * @return true 
     * @return false 
     */
    bool buildTITerms()
    {
        // quadratic objective
        mat<(dim.nu + dim.ny), (dim.nu + dim.ny)> wExtendedState;
        wExtendedState.resize((dim.nu.num() + dim.ny.num()), (dim.nu.num() + dim.ny.num()));
        wExtendedState.setZero();

        mpcProblem.P.setZero();

        for (size_t i = 0; i < (size_t)(dim.ph.num() + 1); i++) {
            wExtendedState.block(0, 0, dim.ny.num(), dim.ny.num()) = wOutput.col(i).asDiagonal();
            wExtendedState.block(dim.ny.num(), dim.ny.num(), dim.nu.num(), dim.nu.num()) = wU.col(i).asDiagonal();

            mpcProblem.P.block(
                i * (dim.nu.num() + dim.nx.num()), i * (dim.nu.num() + dim.nx.num()),
                dim.nu.num() + dim.nx.num(), dim.nu.num() + dim.nx.num())
                = (ssC.transpose() * wExtendedState * ssC);

            // the command increments stop at the last prediction horizon step
            if (i < dim.ph.num()) {
                mpcProblem.P.block(
                    ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (i * dim.nu.num()),
                    ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (i * dim.nu.num()),
                    dim.nu.num(), dim.nu.num())
                    = wDeltaU.col(i).asDiagonal();
            }
        }

        // linear objective dynamics
        mat<((dim.ph + Dim<1>()) * (dim.nu + dim.nx)), (((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + ((dim.ph * dim.nu)))> Aeq;
        Aeq.resize(
            (dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num()),
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + ((dim.ph.num() * dim.nu.num())));

        // build the identity matrices
        mat<(dim.ph + Dim<1>()), (dim.ph + Dim<1>())> augId;
        augId.resize((dim.ph.num() + 1), (dim.ph.num() + 1));
        augId.setZero();
        augId.block(1, 0, dim.ph.num(), dim.ph.num()).setIdentity();

        mat<(dim.ph + Dim<1>()), (dim.ph + Dim<1>())> predHId;
        predHId.resize((dim.ph.num() + 1), (dim.ph.num() + 1));
        predHId.setIdentity();

        mat<(dim.nu + dim.nx), (dim.nu + dim.nx)> extSpaceId;
        extSpaceId.resize((dim.nu.num() + dim.nx.num()), (dim.nu.num() + dim.nx.num()));
        extSpaceId.setIdentity();

        Aeq.block(
            0, 0,
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())))
            = kroneckerProduct(predHId, -extSpaceId).eval() + kroneckerProduct(augId, ssA).eval();

        mat<(dim.ph + Dim<1>()), dim.ph> idenBd;
        idenBd.resize((dim.ph.num() + 1), dim.ph.num());
        idenBd.setZero();
        idenBd.block(1, 0, dim.ph.num(), dim.ph.num()).setIdentity();

        Aeq.block(
            0, ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())), (dim.ph.num() * dim.nu.num()))
            = kroneckerProduct(idenBd, ssB).eval();

        // input, state and output constraints
        mat<(((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (((dim.ph + Dim<1>()) * dim.ny) + (dim.ph * dim.nu))), (((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (dim.ph * dim.nu))> Aineq;
        Aineq.resize(
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num()))),
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (dim.ph.num() * dim.nu.num())));
        Aineq.setZero();

        // add state constraints terms
        Aineq.block(
                 0,
                 0,
                 ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
                 ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())))
            .setIdentity();

        // adding output Cx constraints terms and
        // from the output matrix C we keep only the real system output
        Aineq.block(
            (dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num()),
            0,
            (dim.ph.num() + 1) * dim.ny.num(),
            (dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num()))
            = kroneckerProduct(predHId, ssC.middleRows(0, dim.ny.num())).eval();

        cvec<((dim.ph + Dim<1>()) * (dim.nu + dim.nx))> eMinX, eMaxX;
        eMinX.resize(((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())));
        eMaxX.resize(((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())));

        cvec<dim.nu> tmpMinU, tmpMaxU;
        tmpMinU.resize(dim.nu.num());
        tmpMaxU.resize(dim.nu.num());

        for (size_t i = 0; i < dim.ph.num() + 1; i++) {
            if (i == dim.ph.num()) {
                tmpMinU = minU.col(i - 1);
                tmpMaxU = maxU.col(i - 1);
            } else {
                tmpMinU = minU.col(i);
                tmpMaxU = maxU.col(i);
            }
            eMinX.middleRows(i * (dim.nu.num() + dim.nx.num()), (dim.nu.num() + dim.nx.num())) << minX.col(i), tmpMinU;
            eMaxX.middleRows(i * (dim.nu.num() + dim.nx.num()), (dim.nu.num() + dim.nx.num())) << maxX.col(i), tmpMaxU;
        }

        lineq.middleRows(
            0,
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())))
            = eMinX;

        lineq.middleRows(
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
            ((dim.ph.num() + 1) * dim.ny.num()))
            = Eigen::Map<cvec<((dim.ph + Dim<1>()) * dim.ny)>>(minY.data(), minY.rows() * minY.cols());

        uineq.middleRows(
            0,
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())))
            = eMaxX;

        uineq.middleRows(
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
            ((dim.ph.num() + 1) * dim.ny.num()))
            = Eigen::Map<cvec<((dim.ph + Dim<1>()) * dim.ny)>>(maxY.data(), maxY.rows() * maxY.cols());

        // add more constraints terms
        Aineq.block(
                 ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + ((dim.ph.num() + 1) * dim.ny.num()),
                 ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
                 (dim.ph.num() * dim.nu.num()),
                 (dim.ph.num() * dim.nu.num()))
            .setIdentity();

        // add constraints on delta U to avoid computation
        // of command inputs after the end of the control horizon
        cvec<dim.nu> deltaU;
        deltaU.resize(dim.nu.num());
        deltaU.setOnes();
        double minDeltaU, maxDeltaU;

        for (size_t i = 0; i < dim.ph.num(); i++) {
            minDeltaU = (i > dim.ch.num()) ? 0.0 : -inf;
            maxDeltaU = (i > dim.ch.num()) ? 0.0 : inf;

            lineq.middleRows(
                (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + ((dim.ph.num() + 1) * dim.ny.num())) + (i * dim.nu.num()),
                dim.nu.num())
                = deltaU * minDeltaU;
            uineq.middleRows(
                (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + ((dim.ph.num() + 1) * dim.ny.num())) + (i * dim.nu.num()),
                dim.nu.num())
                = deltaU * maxDeltaU;
        }

        // creation of matrix A
        mpcProblem.A.setZero();

        mpcProblem.A.block(
            0, 0,
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())),
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (dim.ph.num() * dim.nu.num()))
            = Aeq;

        mpcProblem.A.block(
            ((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())), 0,
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (((dim.ph.num() + 1) * dim.ny.num()) + (dim.ph.num() * dim.nu.num()))),
            (((dim.ph.num() + 1) * (dim.nu.num() + dim.nx.num())) + (dim.ph.num() * dim.nu.num())))
            = Aineq;

        return true;
    }

    // the internal state space used is augmented
    // to use the command increments as input of the system
    mat<(dim.nu + dim.nx), (dim.nu + dim.nx)> ssA;
    mat<(dim.nu + dim.nx), dim.nu> ssB;
    mat<(dim.nu + dim.ny), (dim.nu + dim.nx)> ssC;

    // measured disturbances to states and
    // also to the output model
    mat<(dim.nu + dim.nx), dim.ndu> ssBv;
    mat<(dim.nu + dim.ny), dim.ndu> ssDv;

    // objective function weights
    // output, command and delta command
    // tracking error w.r.t reference
    mat<dim.ny, (dim.ph + Dim<1>())> wOutput;
    mat<dim.nu, (dim.ph + Dim<1>())> wU;
    mat<dim.nu, dim.ph> wDeltaU;

    // state/cmd/output constraints
    mat<dim.nx, (dim.ph + Dim<1>())> minX, maxX;
    mat<dim.ny, (dim.ph + Dim<1>())> minY, maxY;
    mat<dim.nu, dim.ph> minU, maxU;

    Problem mpcProblem;
    cvec<((dim.ph + Dim<1>()) * (dim.nu + dim.nx))> leq, ueq;
    cvec<(((dim.ph + Dim<1>()) * (dim.nu + dim.nx)) + (((dim.ph + Dim<1>()) * dim.ny) + (dim.ph * dim.nu)))> lineq, uineq;
};
}