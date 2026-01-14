/*
 *   Copyright (c) 2023-2026 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IOptimizer.hpp>
#include <mpc/LMPC/ProblemBuilder.hpp>

#include <osqp.h>

namespace mpc
{
    /**
     * @brief Linear MPC optimizer interface class
     *
     * @tparam sizer.nx dimension of the state space
     * @tparam sizer.nu dimension of the input space
     * @tparam sizer.ndu dimension of the measured disturbance space
     * @tparam sizer.ny dimension of the output space
     * @tparam Tph length of the prediction horizon
     * @tparam Tch length of the control horizon
     */
    template <MPCSize sizer>
    class LOptimizer : public IOptimizer<sizer>
    {
    private:
        using IComponent<sizer>::checkOrQuit;
        using IDimensionable<sizer>::nu;
        using IDimensionable<sizer>::nx;
        using IDimensionable<sizer>::ndu;
        using IDimensionable<sizer>::ny;
        using IDimensionable<sizer>::ph;
        using IDimensionable<sizer>::ch;
        using IDimensionable<sizer>::ineq;
        using IDimensionable<sizer>::eq;

        using IOptimizer<sizer>::currentSlack;
        using IOptimizer<sizer>::hard;
        using IOptimizer<sizer>::result;
        using IOptimizer<sizer>::sequence;

        LParameters lin_params;

    public:
        LOptimizer() = default;

        ~LOptimizer()
        {
            checkOrQuit();
            clearData();
        }

        /**
         * @brief Initialization hook override. Performing initialization in this
         * method ensures the correct problem dimensions assigment has been
         * already performed
         */
        void onInit() override
        {
            COND_RESIZE_CVEC(sizer,result.cmd, nu());
            result.cmd.setZero();

            COND_RESIZE_MAT(sizer,sequence.state, ph() + 1, nx());
            sequence.state.setZero();
            COND_RESIZE_MAT(sizer,sequence.input, ph() + 1, nu());
            sequence.input.setZero();
            COND_RESIZE_MAT(sizer,sequence.output, ph() + 1, ny());
            sequence.output.setZero();

            COND_RESIZE_MAT(sizer,extInputMeas, ndu(), ph());
            COND_RESIZE_MAT(sizer,outSysRef, ny(), ph());
            COND_RESIZE_MAT(sizer,cmdSysRef, nu(), ph());
            COND_RESIZE_MAT(sizer,deltaCmdSysRef, nu(), ph());

            outSysRef.setZero();
            cmdSysRef.setZero();
            deltaCmdSysRef.setZero();
            extInputMeas.setZero();

            currentSlack = 0;
        }

        /**
         * @brief Set the proble builder
         *
         * @param b optimal problem builder
         */
        void setBuilder(ProblemBuilder<sizer> *b)
        {
            checkOrQuit();
            builder = b;
        }

        /**
         * @brief Set the optmiziation parameters
         *
         * @param param parameters desired
         */
        void setParameters(const Parameters &param) override
        {
            checkOrQuit();
            lin_params = *dynamic_cast<LParameters *>(const_cast<Parameters *>(&param));

            Logger::instance().log(Logger::LogType::DETAIL)
                << "Setting tolerances and stopping criterias"
                << std::endl;
        }

        /**
         * @brief Set the references matrices for the objective function
         *
         * @param outRef reference for the output
         * @param cmdRef reference for the optimal control input
         * @param deltaCmdRef reference for the variation of the optimal control input
         * @return true
         * @return false
         */
        bool setReferences(
            const mat<sizer.ny, sizer.ph> &outRef,
            const mat<sizer.nu, sizer.ph> &cmdRef,
            const mat<sizer.nu, sizer.ph> &deltaCmdRef)
        {
            outSysRef = outRef;
            cmdSysRef = cmdRef;
            deltaCmdSysRef = deltaCmdRef;

            return true;
        }

        /**
         * @brief Set the references vector for the objective function for a specific horizon step
         *
         * @param index index of the horizon step
         * @param outRef reference for the output
         * @param cmdRef reference for the optimal control input
         * @param deltaCmdRef reference for the variation of the optimal control input
         * @return true
         * @return false
         */
        bool setReferences(
            const unsigned int index,
            const cvec<sizer.ny> &outRef,
            const cvec<sizer.nu> &cmdRef,
            const cvec<sizer.nu> &deltaCmdRef)
        {
            outSysRef.col(index) = outRef;
            cmdSysRef.col(index) = cmdRef;
            deltaCmdSysRef.col(index) = deltaCmdRef;

            return true;
        }

        /**
         * @brief Set the exogenous inputs matrix
         *
         * @param uMeas measured exogenous input
         * @return true
         * @return false
         */
        bool setExogenousInputs(const mat<sizer.ndu, sizer.ph> &uMeas)
        {
            extInputMeas = uMeas;
            return true;
        }

        /**
         * @brief Set the exogenous inputs vector for a specific horizon step
         *
         * @param index index of the horizon step
         * @param uMeas measured exogenous input
         * @return true
         * @return false
         */
        bool setExogenousInputs(
            const unsigned int index,
            const cvec<sizer.ndu> &uMeas)
        {
            extInputMeas.col(index) = uMeas;
            return true;
        }

        /**
         * @brief Implementation of the optimization step
         *
         * @param x0 system's variables initial condition
         * @param u0 control action initial condition for warm start
         */
        void run(
            const cvec<sizer.nx> &x0,
            const cvec<sizer.nu> &u0) override
        {
            checkOrQuit();
            Result<sizer.nu> r;

            auto &mpcProblem = builder->get(x0, u0, outSysRef, cmdSysRef, deltaCmdSysRef, extInputMeas);

            smat P, A;
            mpcProblem.getSparse(P, A);

            Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
            Logger::instance().log(Logger::LogType::DETAIL) << "P = " << mpcProblem.P.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "A = " << mpcProblem.A.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "q = " << mpcProblem.q.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "l = " << mpcProblem.l.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "u = " << mpcProblem.u.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::LogType::DETAIL) << "---------------------" << std::endl;

            // getting optimization problem size
            int numVars = P.rows();
            int numConstraints = A.rows();

            // clear and create the problem data struct
            initData();

            // Local CSC matrices for OSQP setup
            OSQPCscMatrix *P_csc = nullptr;
            OSQPCscMatrix *A_csc = nullptr;

            // Build P matrix (upper triangular part is expected by OSQP)
            if (!createOsqpSparseMatrix(P, P_csc))
            {
                Logger::instance().log(Logger::LogType::ERROR) << "Unable to create the P matrix" << std::endl;
            }

            // Build A matrix (full matrix)
            if (!createOsqpSparseMatrix(A, A_csc))
            {
                Logger::instance().log(Logger::LogType::ERROR) << "Unable to create the A matrix" << std::endl;
            }

            // define solver settings as default
            if (settings)
            {
                osqp_set_default_settings(settings);

                settings->alpha = lin_params.alpha;
                settings->verbose = lin_params.verbose ? 1 : 0;
                settings->rho = lin_params.rho;

                // Disable adaptive rho if requested; otherwise keep OSQP default
                if (!lin_params.adaptive_rho)
                {
                    settings->adaptive_rho = OSQP_ADAPTIVE_RHO_UPDATE_DISABLED;
                }
                
                settings->eps_rel = lin_params.eps_rel;
                settings->eps_abs = lin_params.eps_abs;
                settings->eps_prim_inf = lin_params.eps_prim_inf;
                settings->eps_dual_inf = lin_params.eps_dual_inf;
                settings->max_iter = lin_params.maximum_iteration;
                settings->polishing = lin_params.polish ? 1 : 0;
                settings->time_limit = (lin_params.time_limit > 0) ? lin_params.time_limit : OSQP_TIME_LIMIT;
                settings->warm_starting = lin_params.enable_warm_start ? 1 : 0;
            }

            // setup workspace
            exitflag = osqp_setup(&solver,
                                  P_csc,
                                  (const OSQPFloat *)mpcProblem.q.data(),
                                  A_csc,
                                  (const OSQPFloat *)mpcProblem.l.data(),
                                  (const OSQPFloat *)mpcProblem.u.data(),
                                  (OSQPInt)numConstraints,
                                  (OSQPInt)numVars,
                                  settings);

            if (exitflag != 0 || solver == nullptr)
            {
                Logger::instance().log(Logger::LogType::ERROR) << "Unable to setup " << exitflag << std::endl;
            }

            // Free temporary CSC matrices (data has been copied internally by OSQP)
            if (P_csc)
            {
                OSQPCscMatrix_free(P_csc);
                P_csc = nullptr;
            }
            if (A_csc)
            {
                OSQPCscMatrix_free(A_csc);
                A_csc = nullptr;
            }

            // warm starting the solver
            if (solver && settings->warm_starting && optimal_prev_x.size() > 0 && optimal_prev_y.size() > 0)
            {
                exitflag = osqp_warm_start(solver, optimal_prev_x.data(), optimal_prev_y.data());
                if (exitflag != 0)
                {
                    Logger::instance().log(Logger::LogType::ERROR) << "Unable to warm start (code " << exitflag << ")" << std::endl;
                }
            }
            else
            {
                // If warm start is enabled but no previous solution is available, use cold start
                if (solver)
                {
                    osqp_cold_start(solver);
                }
            }

            // solve problem
            exitflag = osqp_solve(solver);
            if (exitflag != 0)
            {
                Logger::instance().log(Logger::LogType::ERROR) << "Unable to solve (code " << exitflag << ")" << std::endl;
            }

            // if the solution is valid update the solution otherwise
            // keep the last feasible solution
            if (solver->solution && solver->solution->x != NULL)
            {
                // storing the previous optimal primal and dual solution to warm start
                optimal_prev_x = std::vector<double>(solver->solution->x, solver->solution->x + numVars);
                optimal_prev_y = std::vector<double>(solver->solution->y, solver->solution->y + numConstraints);

                Logger::instance().log(Logger::LogType::DETAIL) << "Optimal vector: " << std::endl;
                for (size_t i = 0; i < (size_t)P.rows(); i++)
                {
                    Logger::instance().log(Logger::LogType::DETAIL) << solver->solution->x[i] << std::endl;
                }

                // loop over the rows of the optimal sequence
                for (size_t i = 0; i < ph() + 1; i++)
                {
                    // from the extended state vector [x,x_u] we take the first nx entries
                    // to get the optimal sequence of system state
                    for (size_t j = 0; j < nx(); j++)
                    {
                        sequence.state.row(i)[j] = solver->solution->x[i * (nx() + nu()) + j];
                    }

                    // and similarly we take the nu entries to have the optimal sequence of system
                    // input we also needs to deal with the fact that x_u(k) is u(k-1)
                    for (size_t j = nx(); j < nx() + nu(); j++)
                    {
                        // if we are at the end of the horizon we have to
                        if (i + 1 < ph() + 1)
                        {
                            sequence.input.row(i)[j - nx()] = solver->solution->x[(i + 1) * (nx() + nu()) + j];
                        }
                        else
                        {
                            sequence.input.row(i)[j - nx()] = solver->solution->x[i * (nx() + nu()) + j];
                        }
                    }

                    // this just the state mapping together with the optional exogeneous input
                    if (i == 0)
                    {
                        sequence.output.row(i) = builder->mapToOutput(sequence.state.row(i), extInputMeas.col(0));
                    }
                    else
                    {
                        sequence.output.row(i) = builder->mapToOutput(sequence.state.row(i), extInputMeas.col(i - 1));
                    }
                }

                // the optimal command is the first control input in the sequence
                r.cmd = sequence.input.row(0);
                r.solver_status = solver->info->status_val;
                r.cost = solver->info->obj_val;
                r.is_feasible = 
                    solver->info->status_val == OSQP_SOLVED || 
                    solver->info->status_val == OSQP_SOLVED_INACCURATE || 
                    solver->info->status_val == OSQP_MAX_ITER_REACHED;
                
                // convert the return code from the optimizer to the result status
                r.status = convertToResultStatus(r.solver_status);
            }
            else
            {
                // if the solution is not valid we keep the previous solution
                // and we set the return code to -1
                r.cost = mpc::inf;
                r.cmd = result.cmd;
                r.solver_status = -1;
                r.status = ResultStatus::ERROR;

                // in case of invalid solution we ouput all the sequences to zero
                sequence.state.setZero();
                sequence.input.setZero();
                sequence.output.setZero();
            }

            // update the result
            result = r;

            // clear the data to prepare for the next iteration
            clearData();
        }

        // this is a copy of the primal and dual vectors
        // to warm start the solver
        std::vector<double> optimal_prev_x, optimal_prev_y;

    private:
        /**
         * @brief Converts an integer value representing the possible statuses to the corresponding ResultStatus enum value.
         *
         * This function maps the given integer value to the corresponding ResultStatus enum value.
         * If the integer value does not match any known status, the function returns ResultStatus::UNKNOWN.
         *
         * @param status The integer value representing the status.
         * @return The ResultStatus enum value corresponding to the given status integer.
         *
         * @see ResultStatus
         */
        ResultStatus convertToResultStatus(int status)
        {
            switch (status)
            {
            case OSQP_SOLVED:
                return ResultStatus::SUCCESS;
            case OSQP_MAX_ITER_REACHED:
                return ResultStatus::MAX_ITERATION;
            case OSQP_PRIMAL_INFEASIBLE:
                return ResultStatus::INFEASIBLE;
            case OSQP_DUAL_INFEASIBLE:
                return ResultStatus::INFEASIBLE;
            case OSQP_SOLVED_INACCURATE:
                return ResultStatus::SUCCESS;
            case OSQP_PRIMAL_INFEASIBLE_INACCURATE:
                return ResultStatus::SUCCESS;
            case OSQP_DUAL_INFEASIBLE_INACCURATE:
                return ResultStatus::SUCCESS;
            case OSQP_SIGINT:
                return ResultStatus::ERROR;
            case OSQP_TIME_LIMIT_REACHED:
                return ResultStatus::UNKNOWN;
            case OSQP_NON_CVX:
                return ResultStatus::ERROR;
            case OSQP_UNSOLVED:
                return ResultStatus::UNKNOWN;
            default:
                return ResultStatus::UNKNOWN;
            }
        }

        /**
         * @brief Create an osqp sparse matrix from a sparse eigen matrix
         *
         * @param eigenSparseMatrix sparse eigen matrix
         * @param osqpSparseMatrix osqp sparse matrix
         * @return true
         * @return false
         */
        bool createOsqpSparseMatrix(const smat &eigenSparseMatrix, OSQPCscMatrix *&osqpSparseMatrix)
        {
            // Copying into a new sparse matrix to be sure to use a CSC matrix
            // this may perform memory allocation, but this is already the case
            // for allocating the osqpSparseMatrix
            smat colMajorCopy = eigenSparseMatrix;

            // get number of row, columns and nonZeros from Eigen SparseMatrix
            OSQPInt rows = colMajorCopy.rows();
            OSQPInt cols = colMajorCopy.cols();
            OSQPInt numberOfNonZeroCoeff = colMajorCopy.nonZeros();

            // get innerr and outer index
            const int *outerIndexPtr = colMajorCopy.outerIndexPtr();
            const int *innerNonZerosPtr = colMajorCopy.innerNonZeroPtr();

            if (osqpSparseMatrix != nullptr)
            {
                return false;
            }

            // allocate arrays for CSC matrix
            OSQPInt *p = (OSQPInt *)malloc((cols + 1) * sizeof(OSQPInt));
            OSQPInt *i = (OSQPInt *)malloc(numberOfNonZeroCoeff * sizeof(OSQPInt));
            OSQPFloat *x = (OSQPFloat *)malloc(numberOfNonZeroCoeff * sizeof(OSQPFloat));

            if (!p || !i || !x)
            {
                if (p)
                    free(p);
                if (i)
                    free(i);
                if (x)
                    free(x);
                return false;
            }

            int innerOsqpPosition = 0;
            for (int k = 0; k < cols; k++)
            {
                if (colMajorCopy.isCompressed())
                {
                    p[k] = static_cast<OSQPInt>(outerIndexPtr[k]);
                }
                else
                {
                    if (k == 0)
                    {
                        p[k] = 0;
                    }
                    else
                    {
                        p[k] = p[k - 1] + innerNonZerosPtr[k - 1];
                    }
                }
                for (typename smat::InnerIterator it(colMajorCopy, k); it; ++it)
                {
                    i[innerOsqpPosition] = static_cast<OSQPInt>(it.row());
                    x[innerOsqpPosition] = static_cast<OSQPFloat>(it.value());
                    innerOsqpPosition++;
                }
            }

            p[static_cast<int>(cols)] = static_cast<OSQPInt>(innerOsqpPosition);
            assert(innerOsqpPosition == numberOfNonZeroCoeff);

            // wrap arrays into OSQP CSC matrix (owned by wrapper for automatic free)
            osqpSparseMatrix = OSQPCscMatrix_new(rows, cols, numberOfNonZeroCoeff, x, i, p);
            if (!osqpSparseMatrix)
            {
                free(p);
                free(i);
                free(x);
                return false;
            }
            osqpSparseMatrix->owned = 1;
            return true;
        }

        /**
         * @brief Clear the current allocated osqp problem data structures
         */
        void clearData()
        {
            osqp_cleanup(solver);

            if (settings)
            {
                OSQPSettings_free(settings);
                settings = nullptr;
            }
        }

        /**
         * @brief Initialize the osqp problem data structures
         */
        void initData()
        {
            settings = OSQPSettings_new();
        }

        OSQPSolver *solver;
        OSQPSettings *settings;
        OSQPInt exitflag = 0;

        mat<sizer.ny, sizer.ph> outSysRef;
        mat<sizer.nu, sizer.ph> cmdSysRef, deltaCmdSysRef;
        mat<sizer.ndu, sizer.ph> extInputMeas;

        ProblemBuilder<sizer> *builder;
    };
} // namespace mpc
