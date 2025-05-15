/*
 *   Copyright (c) 2023-2025 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IComponent.hpp>
#include <mpc/Utils.hpp>
#include <unsupported/Eigen/KroneckerProduct>

namespace mpc
{
    /**
     * @brief Linear MPC optimal problem builder
     *
     * @tparam sizer.nx dimension of the state space
     * @tparam sizer.nu dimension of the input space
     * @tparam sizer.ndu dimension of the measured disturbance space
     * @tparam sizer.ny dimension of the output space
     * @tparam sizer.ph length of the prediction horizon
     * @tparam Tch length of the control horizon
     */
    template <MPCSize sizer>
    class ProblemBuilder : public IComponent<sizer>
    {
    private:
        using IComponent<sizer>::checkOrQuit;
        using IComponent<sizer>::nu;
        using IComponent<sizer>::nx;
        using IComponent<sizer>::ndu;
        using IComponent<sizer>::ny;
        using IComponent<sizer>::ph;
        using IComponent<sizer>::ch;
        using IComponent<sizer>::ineq;
        using IComponent<sizer>::eq;

    public:
        /**
         * @brief Linear optimal problem
         */
        class Problem
        {
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
            void getSparse(smat &Psparse, smat &Asparse) const
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
            mat<(((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (sizer.ph * sizer.nu)), (((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (sizer.ph * sizer.nu))> P;
            // objective_vector is q
            cvec<(((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (sizer.ph * sizer.nu))> q;
            // constraint_matrix is A
            mat<(((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (((sizer.ph + 1) * sizer.ny) + (sizer.ph * sizer.nu))) + (sizer.ph + 1)), (((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (sizer.ph * sizer.nu))> A;
            // lower_bounds is l and upper_bounds is u
            cvec<(((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (((sizer.ph + 1) * sizer.ny) + (sizer.ph * sizer.nu))) + (sizer.ph + 1))> l, u;
        };

        ProblemBuilder() = default;
        ~ProblemBuilder() = default;

        /**
         * @brief Initialization hook override used to perform the
         * initialization procedure. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed.
         */
        void onInit() override
        {
            COND_RESIZE_MAT(sizer,ssA, nu() + nx(), nu() + nx());
            COND_RESIZE_MAT(sizer,ssB, nu() + nx(), nu());
            COND_RESIZE_MAT(sizer,ssC, nu() + ny(), nu() + nx());
            COND_RESIZE_MAT(sizer,ssBv, nu() + nx(), ndu());
            COND_RESIZE_MAT(sizer,ssDv, nu() + ny(), ndu());

            COND_RESIZE_MAT(sizer,wOutput, ny(), (ph() + 1));
            COND_RESIZE_MAT(sizer,wU, nu(), (ph() + 1));
            COND_RESIZE_MAT(sizer,wDeltaU, nu(), ph());

            COND_RESIZE_MAT(sizer,minX, nx(), (ph() + 1));
            COND_RESIZE_MAT(sizer,maxX, nx(), (ph() + 1));

            COND_RESIZE_MAT(sizer,minY, ny(), (ph() + 1));
            COND_RESIZE_MAT(sizer,maxY, ny(), (ph() + 1));

            COND_RESIZE_MAT(sizer,minU, nu(), ph());
            COND_RESIZE_MAT(sizer,maxU, nu(), ph());

            COND_RESIZE_CVEC(sizer,sMin, ph() + 1);
            COND_RESIZE_CVEC(sizer,sMax, ph() + 1);
            COND_RESIZE_MAT(sizer,sMultiplier, (ph() + 1), (ph() + 1) * (nu() + nx()));

            COND_RESIZE_CVEC(sizer,leq, ((ph() + 1) * (nu() + nx())));
            COND_RESIZE_CVEC(sizer,ueq, ((ph() + 1) * (nu() + nx())));

            COND_RESIZE_CVEC(sizer,lineq, (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu()) + (ph() + 1))));
            COND_RESIZE_CVEC(sizer,uineq, (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu()) + (ph() + 1))));
            COND_RESIZE_CVEC(sizer,ineq_offset, (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu()) + (ph() + 1))));

            ssA.setZero();
            ssB.setZero();
            ssC.setZero();
            ssBv.setZero();
            ssDv.setZero();

            wOutput.setZero();
            wU.setZero();
            wDeltaU.setZero();

            // setting the box constraints to -inf and inf
            minX.setConstant(-inf);
            minY.setConstant(-inf);
            minU.setConstant(-inf);
            maxX.setConstant(inf);
            maxY.setConstant(inf);
            maxU.setConstant(inf);

            // setting the scalar constraints to -inf and inf
            sMin.setConstant(-inf);
            sMax.setConstant(inf);

            // multiplier is defaulted to zero
            sMultiplier.setZero();

            leq.setZero();
            ueq.setZero();

            lineq.setZero();
            uineq.setZero();

            COND_RESIZE_MAT(sizer,mpcProblem.P,
                            (((ph() + 1) * (nu() + nx())) + (ph() * nu())),
                            (((ph() + 1) * (nu() + nx())) + (ph() * nu())));
            COND_RESIZE_CVEC(sizer,mpcProblem.q,
                             (((ph() + 1) * (nu() + nx())) + (ph() * nu())));
            COND_RESIZE_MAT(sizer,mpcProblem.A,
                            (((ph() + 1) * (nu() + nx())) + ((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu()) + (ph() + 1))),
                            (((ph() + 1) * (nu() + nx())) + (ph() * nu())));
            COND_RESIZE_CVEC(sizer,mpcProblem.l,
                             (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu())) + (ph() + 1))));
            COND_RESIZE_CVEC(sizer,mpcProblem.u,
                             (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu())) + (ph() + 1))));

            mpcProblem.P.setZero();
            mpcProblem.q.setZero();
            mpcProblem.A.setZero();
            mpcProblem.l.setZero();
            mpcProblem.u.setZero();

            // let's build the time invariant terms using the default conditions
            buildTimeInvariantTems();
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
            const mat<sizer.nx, sizer.nx> &A, const mat<sizer.nx, sizer.nu> &B,
            const mat<sizer.ny, sizer.nx> &C)
        {
            checkOrQuit();

            // state vector [x(t) x_u(t)], where x_u(t) = u(t-1)
            // we are augmenting the system to store the command input of the current timestep
            // the system we are using is the following:
            // x(t + 1) = A x(t) + B u(t-1) + B Delta_u(t)
            // x_u(t + 1) = x_u(t) + Delta_u(t)

            ssA.block(0, 0, nx(), nx()) = A;
            ssA.block(0, nx(), nx(), nu()) = B;
            ssA.block(nx(), 0, nu(), nx()).setZero();
            ssA.block(nx(), nx(), nu(), nu()).setIdentity();

            ssB.block(0, 0, nx(), nu()) = B;
            ssB.block(nx(), 0, nu(), nu()).setIdentity();

            // we put on the output also the command to allow its penalization
            // NOTE: here we have that at horizon step p we have in output the
            // command applied at the step p-1
            ssC.block(0, 0, ny(), nx()) = C;
            ssC.block(ny(), nx(), nu(), nu()).setIdentity();

            return buildTimeInvariantTems();
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
        bool setExogenousInput(
            const mat<sizer.nx, sizer.ndu> &Bd,
            const mat<sizer.ny, sizer.ndu> &Dd)
        {
            checkOrQuit();

            // the exogenous inputs goes only to states and outputs
            ssBv.setZero();
            ssBv.block(0, 0, nx(), ndu()) = Bd;

            ssDv.setZero();
            ssDv.block(0, 0, ny(), ndu()) = Dd;

            return buildTimeInvariantTems();
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
            const mat<sizer.ny, sizer.ph> &OWeight,
            const mat<sizer.nu, sizer.ph> &UWeight,
            const mat<sizer.nu, sizer.ph> &DeltaUWeight)
        {
            checkOrQuit();

            wOutput.block(0, 1, ny(), ph()) = OWeight;
            wOutput.col(0) = OWeight.col(0);

            wU.block(0, 1, nu(), ph()) = UWeight;
            wU.col(0) = UWeight.col(0);

            wDeltaU = DeltaUWeight;

            return buildTimeInvariantTems();
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
            const unsigned int &index,
            const cvec<sizer.ny> &OWeight,
            const cvec<sizer.nu> &UWeight,
            const cvec<sizer.nu> &DeltaUWeight)
        {
            checkOrQuit();

            wOutput.block(0, index + 1, ny(), 1) = OWeight;
            if (index == 0)
            {
                wOutput.col(0) = OWeight;
            }

            wU.block(0, index + 1, nu(), 1) = UWeight;
            if (index == 0)
            {
                wU.col(0) = UWeight;
            }

            wDeltaU.block(0, index, nu(), 1) = DeltaUWeight;

            return buildTimeInvariantTems();
        }

        /**
         * @brief Set the scalar constraint for a specific horizon step
         *
         * @param index index of the horizon step
         * @param min the lower bound
         * @param max the upper bound
         * @param X the constant term multiplied to the state
         * @param U the constant term multiplied to the input
         * @return true
         * @return false
         */
        bool setScalarConstraint(
            const unsigned index,
            const double &min, const double &max,
            const cvec<sizer.nx> &X, const cvec<sizer.nu> &U)
        {
            checkOrQuit();

            sMin[index + 1] = min;
            if (index == 0)
            {
                sMin(0) = min;
            }

            sMax[index + 1] = max;
            if (index == 0)
            {
                sMax(0) = max;
            }

            for (size_t i = 0; i < ph() + 1; i++)
            {
                sMultiplier.block(i, i * (nx() + nu()), 1, nx() + nu()) << X.transpose(), U.transpose();
            }

            return buildTimeInvariantTems();
        }

        /**
         * @brief Set the scalar constraint
         *
         * @param MinMat the lower bound
         * @param MaxMat the upper bound
         * @param X the constant term multiplied to the state
         * @param U the constant term multiplied to the input
         * @return true
         * @return false
         */
        bool setScalarConstraint(
            const cvec<sizer.ph> &MinMat, const cvec<sizer.ph> &MaxMat,
            const cvec<sizer.nx> &X, const cvec<sizer.nu> &U)
        {
            checkOrQuit();

            sMin.segment(1, ph()) = MinMat;
            sMin(0) = MinMat(0);

            sMax.segment(1, ph()) = MaxMat;
            sMax(0) = MaxMat(0);

            for (size_t i = 0; i < ph() + 1; i++)
            {
                sMultiplier.block(i, i * (nx() + nu()), 1, nx() + nu()) << X.transpose(), U.transpose();
            }

            return buildTimeInvariantTems();
        }

        /**
         * @brief Sets the state bounds for the LMPC problem.
         *
         * This function sets the lower and upper bounds for the state variables of the LMPC problem.
         * The state bounds are specified as matrices, where each column represents the bounds for a specific time step.
         * The first column of the matrices represents the initial state bounds.
         *
         * @param XMinMat The matrix specifying the lower bounds for the state variables.
         * @param XMaxMat The matrix specifying the upper bounds for the state variables.
         * @return True if the time-invariant terms are successfully built, false otherwise.
         */
        bool setStateBounds(const mat<sizer.nx, sizer.ph> XMinMat, const mat<sizer.nx, sizer.ph> XMaxMat)
        {
            checkOrQuit();

            minX.block(0, 1, nx(), ph()) = XMinMat;
            minX.col(0) = XMinMat.col(0);
            maxX.block(0, 1, nx(), ph()) = XMaxMat;
            maxX.col(0) = XMaxMat.col(0);

            return buildTimeInvariantTems();
        }

        /**
         * Sets the input bounds for the LMPC problem.
         *
         * @param UMinMat The lower bound matrix for the inputs.
         * @param UMaxMat The upper bound matrix for the inputs.
         * @return True if the time-invariant terms are successfully built, false otherwise.
         */
        bool setInputBounds(const mat<sizer.nu, sizer.ch> UMinMat, const mat<sizer.nu, sizer.ch> UMaxMat)
        {
            checkOrQuit();

            // fill the portion of the prediction horizon with the input bounds on the control horizon
            minU.block(0, 0, nu(), ch()) = UMinMat;
            maxU.block(0, 0, nu(), ch()) = UMaxMat;

            // then the last bounds of the control horizon are used to fill the remaining part of the prediction horizon
            if (ch() < ph())
            {
                minU.block(0, ch(), nu(), ph() - ch()) = UMinMat.col(ch() - 1).replicate(1, ph() - ch());
                maxU.block(0, ch(), nu(), ph() - ch()) = UMaxMat.col(ch() - 1).replicate(1, ph() - ch());
            }

            return buildTimeInvariantTems();
        }

        /**
         * Sets the output bounds for the LMPC problem.
         *
         * @param YMinMat The matrix representing the lower bounds for the output variables.
         * @param YMaxMat The matrix representing the upper bounds for the output variables.
         * @return True if the time-invariant terms are successfully built, false otherwise.
         */
        bool setOutputBounds(const mat<sizer.ny, sizer.ph> YMinMat, const mat<sizer.ny, sizer.ph> YMaxMat)
        {
            checkOrQuit();

            minY.block(0, 1, ny(), ph()) = YMinMat;
            minY.col(0) = YMinMat.col(0);
            maxY.block(0, 1, ny(), ph()) = YMaxMat;
            maxY.col(0) = YMaxMat.col(0);

            return buildTimeInvariantTems();
        }

        /**
         * Sets the state bounds for a specific index.
         *
         * @param index The index of the state bounds to set.
         * @param XMin The minimum values for the state bounds.
         * @param XMax The maximum values for the state bounds.
         * @return True if the state bounds were successfully set, false otherwise.
         */
        bool setStateBounds(const unsigned int &index, const cvec<sizer.nx> XMin, const cvec<sizer.nx> XMax)
        {
            checkOrQuit();

            minX.block(0, index + 1, nx(), 1) = XMin;
            if (index == 0)
            {
                minX.col(0) = XMin;
            }

            maxX.block(0, index + 1, nx(), 1) = XMax;
            if (index == 0)
            {
                maxX.col(0) = XMax;
            }

            return buildTimeInvariantTems();
        }

        /**
         * Sets the input bounds for a specific index.
         *
         * @param index The index of the input bounds to set.
         * @param UMin The lower bounds for the input.
         * @param UMax The upper bounds for the input.
         * @return True if the input bounds were set successfully, false otherwise.
         */
        bool setInputBounds(const unsigned int &index, const cvec<sizer.nu> UMin, const cvec<sizer.nu> UMax)
        {
            checkOrQuit();

            minU.block(0, index, nu(), 1) = UMin;
            maxU.block(0, index, nu(), 1) = UMax;

            return buildTimeInvariantTems();
        }

        /**
         * Sets the output bounds for a specific index.
         *
         * @param index The index of the output bounds to set.
         * @param YMin The minimum values for the output bounds.
         * @param YMax The maximum values for the output bounds.
         * @return True if the output bounds were successfully set, false otherwise.
         */
        bool setOutputBounds(const unsigned int &index, const cvec<sizer.ny> YMin, const cvec<sizer.ny> YMax)
        {
            checkOrQuit();

            minY.block(0, index + 1, ny(), 1) = YMin;
            if (index == 0)
            {
                minY.col(0) = YMin;
            }

            maxY.block(0, index + 1, ny(), 1) = YMax;
            if (index == 0)
            {
                maxY.col(0) = YMax;
            }

            return buildTimeInvariantTems();
        }

        /**
         * @brief Compute the relative output of the system based on the desired
         * state and measured disturbance
         *
         * @param desState desired state vector to project
         * @param measDist measured disturbance vector
         * @return cvec<sizer.ny> the output vector
         */
        cvec<sizer.ny> mapToOutput(const cvec<sizer.nx> &desState, const cvec<sizer.ndu> &measDist)
        {
            return ssC.block(0, 0, ny(), nx()) * desState + ssDv.block(0, 0, ny(), ndu()) * measDist;
        }

        /**
         * @brief Request the generation of a new MPC optimization problem
         *
         * @param x0 initial condition of the system's dynamics vector
         * @param yRef output reference matrix
         * @param uRef control input reference matrix
         * @param deltaURef control input variation reference matrix
         * @param uMeas external disturbances matrix
         */
        const Problem &get(
            const cvec<sizer.nx> &x0,
            const cvec<sizer.nu> &u0,
            const mat<sizer.ny, sizer.ph> &yRef,
            const mat<sizer.nu, sizer.ph> &uRef,
            const mat<sizer.nu, sizer.ph> &deltaURef,
            const mat<sizer.ndu, sizer.ph> &uMeas)
        {
            // linear objective terms must be computed at each control loop since
            // it depends on the references and the refs can changes over time
            mat<(sizer.ny + sizer.nu), (sizer.ny + sizer.nu)> wExtendedState;
            COND_RESIZE_MAT(sizer,wExtendedState, (ny() + nu()), (ny() + nu()));
            wExtendedState.setZero();

            cvec<(sizer.ny + sizer.nu)> eRef;
            COND_RESIZE_CVEC(sizer,eRef, ny() + nu());

            mpcProblem.q.setZero();
            leq.setZero();

            ineq_offset.setZero();

            // creation of bounds and here is the right place to take into account
            // of the measured exogenous inputs on the output (we are gonna treat them as offsets)
            mpcProblem.l.setZero();
            mpcProblem.u.setZero();

            for (size_t i = 0; i < ph() + 1; i++)
            {
                // definition of the references (this check is needed since at the first
                // step of the prediction horizon there is the current state of the system)
                cvec<sizer.ny> yRef_ex;
                cvec<sizer.nu> uRef_ex;
                cvec<sizer.nu> deltaURef_ex;
                cvec<sizer.ndu> uMeas_ex;

                if (i == 0)
                {
                    yRef_ex = yRef.col(0);
                    uRef_ex = uRef.col(0);
                    deltaURef_ex = deltaURef.col(0);
                    uMeas_ex = uMeas.col(0);
                }
                else
                {
                    yRef_ex = yRef.col(i - 1);
                    uRef_ex = uRef.col(i - 1);
                    deltaURef_ex = deltaURef.col(i - 1);
                    uMeas_ex = uMeas.col(i - 1);
                }

                eRef << yRef_ex, uRef_ex;

                wExtendedState.block(0, 0, ny(), ny()) = wOutput.col(i).asDiagonal();
                wExtendedState.block(
                    ny(), ny(),
                    nu(), nu()) = wU.col(i).asDiagonal();

                mpcProblem.q.middleRows(
                    i * (nx() + nu()), nx() + nu()) = ssC.transpose() * wExtendedState * (-eRef + (ssDv * uMeas_ex));

                // the command increments stop at the last prediction horizon step
                if (i < ph())
                {
                    mpcProblem.q.middleRows(
                        ((ph() + 1) * (nu() + nx())) + (i * nu()),
                        nu()) = -(wDeltaU.col(i).asDiagonal() * deltaURef_ex);
                }

                // the first entry of the state evolution is the initial condition of the states
                if (i > 0)
                {
                    leq.middleRows(i * (nx() + nu()), nx() + nu()) = -ssBv * uMeas_ex;
                }

                // let's add on the output part of the system
                // any contribution of the exogenous inputs on the output function
                ineq_offset.middleRows(
                    (i * ny()) + ((ph() + 1) * (nu() + nx())),
                    ny()) = -ssDv.block(0, 0, ny(), ndu()) * uMeas_ex;
            }

            // state evolution depends on the initial condition and
            // on the exogeneous inputs so they are changes over time
            leq.middleRows(0, nx()) = -x0;
            leq.middleRows(nx(), nu()) = -u0;

            // set lower and upper bound equal in order to have equality constraints
            ueq = leq;

            mpcProblem.l.middleRows(
                0, (ph() + 1) * (nu() + nx())) = leq;

            mpcProblem.u.middleRows(
                0, (ph() + 1) * (nu() + nx())) = ueq;

            mpcProblem.l.middleRows(
                (ph() + 1) * (nu() + nx()),
                ((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny()) + (ph() * nu()) + (ph() + 1)) = lineq + ineq_offset;

            mpcProblem.u.middleRows(
                (ph() + 1) * (nu() + nx()),
                ((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny()) + (ph() * nu()) + (ph() + 1)) = uineq + ineq_offset;

            return mpcProblem;
        }

    private:
        /**
         * @brief Build the time invariant optimal control problem terms
         *
         * @return true
         * @return false
         */
        bool buildTimeInvariantTems()
        {
            // quadratic objective
            mat<(sizer.nu + sizer.ny), (sizer.nu + sizer.ny)> wExtendedState;
            COND_RESIZE_MAT(sizer,wExtendedState, (nu() + ny()), (nu() + ny()));
            wExtendedState.setZero();

            mpcProblem.P.setZero();

            for (size_t i = 0; i < (size_t)(ph() + 1); i++)
            {
                wExtendedState.block(0, 0, ny(), ny()) = wOutput.col(i).asDiagonal();
                wExtendedState.block(ny(), ny(), nu(), nu()) = wU.col(i).asDiagonal();

                mpcProblem.P.block(
                    i * (nu() + nx()), i * (nu() + nx()),
                    nu() + nx(), nu() + nx()) = (ssC.transpose() * wExtendedState * ssC);

                // the command increments stop at the last prediction horizon step
                if (i < ph())
                {
                    mpcProblem.P.block(
                        ((ph() + 1) * (nu() + nx())) + (i * nu()),
                        ((ph() + 1) * (nu() + nx())) + (i * nu()),
                        nu(), nu()) = wDeltaU.col(i).asDiagonal();
                }
            }

            // linear objective dynamics
            mat<((sizer.ph + 1) * (sizer.nu + sizer.nx)), (((sizer.ph + 1) * (sizer.nu + sizer.nx)) + ((sizer.ph * sizer.nu)))> Aeq;
            COND_RESIZE_MAT(sizer,Aeq,
                            (ph() + 1) * (nu() + nx()),
                            ((ph() + 1) * (nu() + nx())) + ((ph() * nu())));

            // build the identity matrices
            mat<sizer.ph + 1, sizer.ph + 1> augId;
            COND_RESIZE_MAT(sizer,augId, (ph() + 1), (ph() + 1));
            augId.setZero();
            augId.block(1, 0, ph(), ph()).setIdentity();

            mat<sizer.ph + 1, sizer.ph + 1> predHId;
            COND_RESIZE_MAT(sizer,predHId, (ph() + 1), (ph() + 1));
            predHId.setIdentity();

            mat<(sizer.nu + sizer.nx), (sizer.nu + sizer.nx)> extSpaceId;
            COND_RESIZE_MAT(sizer,extSpaceId, (nu() + nx()), (nu() + nx()));
            extSpaceId.setIdentity();

            Aeq.block(
                0, 0,
                ((ph() + 1) * (nu() + nx())),
                ((ph() + 1) * (nu() + nx()))) = kroneckerProduct(predHId, -extSpaceId).eval() + kroneckerProduct(augId, ssA).eval();

            mat<sizer.ph + 1, sizer.ph> idenBd;
            COND_RESIZE_MAT(sizer,idenBd, (ph() + 1), ph());
            idenBd.setZero();
            idenBd.block(1, 0, ph(), ph()).setIdentity();

            Aeq.block(
                0, ((ph() + 1) * (nu() + nx())),
                ((ph() + 1) * (nu() + nx())), (ph() * nu())) = kroneckerProduct(idenBd, ssB).eval();

            // input, state and output constraints
            mat<(((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (((sizer.ph + 1) * sizer.ny) + (sizer.ph * sizer.nu)) + (sizer.ph + 1)), (((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (sizer.ph * sizer.nu))> Aineq;
            COND_RESIZE_MAT(sizer,Aineq,
                            (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu()) + (ph() + 1))),
                            (((ph() + 1) * (nu() + nx())) + (ph() * nu())));
            Aineq.setZero();

            // add state constraints terms
            Aineq.block(
                     0,
                     0,
                     ((ph() + 1) * (nu() + nx())),
                     ((ph() + 1) * (nu() + nx())))
                .setIdentity();

            // adding output Cx constraints terms and
            // from the output matrix C we keep only the real system output
            Aineq.block(
                (ph() + 1) * (nu() + nx()),
                0,
                (ph() + 1) * ny(),
                (ph() + 1) * (nu() + nx())) = kroneckerProduct(predHId, ssC.middleRows(0, ny())).eval();

            cvec<((sizer.ph + 1) * (sizer.nu + sizer.nx))> eMinX, eMaxX;
            COND_RESIZE_CVEC(sizer,eMinX, ((ph() + 1) * (nu() + nx())));
            COND_RESIZE_CVEC(sizer,eMaxX, ((ph() + 1) * (nu() + nx())));

            cvec<sizer.nu> tmpMinU, tmpMaxU;
            COND_RESIZE_CVEC(sizer,tmpMinU, nu());
            COND_RESIZE_CVEC(sizer,tmpMaxU, nu());

            for (size_t i = 0; i < ph() + 1; i++)
            {
                if (i == ph())
                {
                    tmpMinU = minU.col(i - 1);
                    tmpMaxU = maxU.col(i - 1);
                }
                else
                {
                    tmpMinU = minU.col(i);
                    tmpMaxU = maxU.col(i);
                }
                eMinX.middleRows(i * (nu() + nx()), (nu() + nx())) << minX.col(i), tmpMinU;
                eMaxX.middleRows(i * (nu() + nx()), (nu() + nx())) << maxX.col(i), tmpMaxU;
            }

            lineq.middleRows(
                0,
                ((ph() + 1) * (nu() + nx()))) = eMinX;

            lineq.middleRows(
                ((ph() + 1) * (nu() + nx())),
                ((ph() + 1) * ny())) = Eigen::Map<cvec<((sizer.ph + 1) * sizer.ny)>>(minY.data(), minY.rows() * minY.cols());

            uineq.middleRows(
                0,
                ((ph() + 1) * (nu() + nx()))) = eMaxX;

            uineq.middleRows(
                ((ph() + 1) * (nu() + nx())),
                ((ph() + 1) * ny())) = Eigen::Map<cvec<((sizer.ph + 1) * sizer.ny)>>(maxY.data(), maxY.rows() * maxY.cols());

            // add more constraints terms
            Aineq.block(
                     ((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny()),
                     ((ph() + 1) * (nu() + nx())),
                     (ph() * nu()),
                     (ph() * nu()))
                .setIdentity();

            // add constraints on delta U to avoid computation
            // of command inputs after the end of the control horizon
            cvec<sizer.nu> deltaU;
            COND_RESIZE_CVEC(sizer,deltaU, nu());
            deltaU.setOnes();
            double minDeltaU, maxDeltaU;

            for (size_t i = 0; i < ph(); i++)
            {
                minDeltaU = (i > ch()) ? 0.0 : -inf;
                maxDeltaU = (i > ch()) ? 0.0 : inf;

                lineq.middleRows(
                    (((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny())) + (i * nu()),
                    nu()) = deltaU * minDeltaU;
                uineq.middleRows(
                    (((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny())) + (i * nu()),
                    nu()) = deltaU * maxDeltaU;
            }

            // insert a scalar constraint
            // TODO add support for multiple scalar constraints
            Aineq.block(
                ((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny()) + (ph() * nu()),
                0,
                ph() + 1,
                (ph() + 1) * (nu() + nx())) = sMultiplier;

            lineq.middleRows(
                ((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny()) + (ph() * nu()),
                ph() + 1) = sMin;

            uineq.middleRows(
                ((ph() + 1) * (nu() + nx())) + ((ph() + 1) * ny()) + (ph() * nu()),
                ph() + 1) = sMax;

            // creation of matrix A
            mpcProblem.A.setZero();

            mpcProblem.A.block(
                0, 0,
                ((ph() + 1) * (nu() + nx())),
                ((ph() + 1) * (nu() + nx())) + (ph() * nu())) = Aeq;

            mpcProblem.A.block(
                ((ph() + 1) * (nu() + nx())), 0,
                (((ph() + 1) * (nu() + nx())) + (((ph() + 1) * ny()) + (ph() * nu())) + (ph() + 1)),
                (((ph() + 1) * (nu() + nx())) + (ph() * nu()))) = Aineq;

            return true;
        }

        // the internal state space used is augmented
        // to use the command increments as input of the system
        mat<(sizer.nu + sizer.nx), (sizer.nu + sizer.nx)> ssA;
        mat<(sizer.nu + sizer.nx), sizer.nu> ssB;
        mat<(sizer.nu + sizer.ny), (sizer.nu + sizer.nx)> ssC;

        // measured disturbances to states and
        // also to the output model
        mat<(sizer.nu + sizer.nx), sizer.ndu> ssBv;
        mat<(sizer.nu + sizer.ny), sizer.ndu> ssDv;

        // objective function weights
        // output, command and delta command
        // tracking error w.r.t reference
        mat<sizer.ny, sizer.ph + 1> wOutput;
        mat<sizer.nu, sizer.ph + 1> wU;
        mat<sizer.nu, sizer.ph> wDeltaU;

        // state/cmd/output constraints
        mat<sizer.nx, sizer.ph + 1> minX, maxX;
        mat<sizer.ny, sizer.ph + 1> minY, maxY;
        mat<sizer.nu, sizer.ph> minU, maxU;

        // scalar constraint
        cvec<sizer.ph + 1> sMin;
        cvec<sizer.ph + 1> sMax;
        mat<sizer.ph + 1, (sizer.ph + 1) * (sizer.nx + sizer.nu)> sMultiplier;

        Problem mpcProblem;
        cvec<((sizer.ph + 1) * (sizer.nu + sizer.nx))> leq, ueq;
        cvec<(((sizer.ph + 1) * (sizer.nu + sizer.nx)) + (((sizer.ph + 1) * sizer.ny) + (sizer.ph * sizer.nu)) + (sizer.ph + 1))> lineq, uineq, ineq_offset;
    };
}