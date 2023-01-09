/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <mpc/IOptimizer.hpp>
#include <mpc/LMPC/ProblemBuilder.hpp>

#include <osqp/osqp.h>

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
        void onInit()
        {
            result.cmd.resize(nu());
            result.cmd.setZero();

            sequence.state.resize(ph(), nx());
            sequence.state.setZero();
            sequence.input.resize(ph(), nu());
            sequence.input.setZero();
            sequence.output.resize(ph(), ny());
            sequence.output.setZero();

            extInputMeas.resize(ndu(), ph());
            outSysRef.resize(ny(), ph());
            cmdSysRef.resize(nu(), ph());
            deltaCmdSysRef.resize(nu(), ph());

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
        void setParameters(const Parameters &param)
        {
            checkOrQuit();
            lin_params = *dynamic_cast<LParameters *>(const_cast<Parameters *>(&param));

            Logger::instance().log(Logger::log_type::DETAIL)
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
         * @brief Set the exogenuos inputs matrix
         *
         * @param uMeas measured exogenuos input
         * @return true
         * @return false
         */
        bool setExogenuosInputs(const mat<sizer.ndu, sizer.ph> &uMeas)
        {
            extInputMeas = uMeas;
            return true;
        }

        /**
         * @brief Set the exogenuos inputs vector for a specific horizon step
         *
         * @param index index of the horizon step
         * @param uMeas measured exogenuos input
         * @return true
         * @return false
         */
        bool setExogenuosInputs(
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
            const cvec<sizer.nu> &u0)
        {
            checkOrQuit();
            Result<sizer.nu> r;

            auto &mpcProblem = builder->get(x0, u0, outSysRef, cmdSysRef, deltaCmdSysRef, extInputMeas);

            smat P, A;
            mpcProblem.getSparse(P, A);

            Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
            Logger::instance().log(Logger::log_type::DETAIL) << "P = " << mpcProblem.P.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "A = " << mpcProblem.A.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "q = " << mpcProblem.q.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "l = " << mpcProblem.l.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "---------------------" << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "u = " << mpcProblem.u.format(OctaveFmt) << std::endl;
            Logger::instance().log(Logger::log_type::DETAIL) << "---------------------" << std::endl;

            // getting optimization problem size
            int numVars = P.rows();
            int numConstraints = A.rows();

            // clear and create the problem data struct
            initData();

            if (data)
            {
                data->n = numVars;
                data->m = numConstraints;

                if (!createOsqpSparseMatrix(P, data->P))
                {
                    Logger::instance().log(Logger::log_type::ERROR) << "Unable to create the P matrix" << std::endl;
                }

                data->q = (c_float *)mpcProblem.q.data();

                if (!createOsqpSparseMatrix(A, data->A))
                {
                    Logger::instance().log(Logger::log_type::ERROR) << "Unable to create the A matrix" << std::endl;
                }

                data->l = (c_float *)mpcProblem.l.data();
                data->u = (c_float *)mpcProblem.u.data();
            }

            // define solver settings as default
            if (settings)
            {
                osqp_set_default_settings(settings);

                settings->alpha = lin_params.alpha;
                settings->verbose = lin_params.verbose ? 1 : 0;
                settings->rho = lin_params.rho;
                settings->adaptive_rho = lin_params.adaptive_rho ? 1 : 0;
                settings->eps_rel = lin_params.eps_rel;
                settings->eps_abs = lin_params.eps_abs;
                settings->eps_prim_inf = lin_params.eps_prim_inf;
                settings->eps_dual_inf = lin_params.eps_dual_inf;
                settings->max_iter = lin_params.maximum_iteration;
                settings->polish = lin_params.polish ? 1 : 0;
            }

            // setup workspace
            exitflag = osqp_setup(&work, data, settings);
            if (exitflag > 0)
            {
                Logger::instance().log(Logger::log_type::ERROR) << "Unable to solve " << exitflag << std::endl;
            }

            // solve problem
            osqp_solve(work);

            // if the solution is valid update the solution otherwise
            // keep the last feasible solution
            if (work->solution->x != NULL)
            {
                int index = 0;

                cvec<sizer.nu> optCmd;
                optCmd.resize(nu());

                for (size_t i = ((ph() + 1) * (nx() + nu())); i < (((ph() + 1) * (nx() + nu())) + nu()); i++)
                {
                    optCmd[index++] = work->solution->x[i];
                }

                r.cmd = optCmd;
                r.retcode = work->info->status_val;
                r.cost = work->info->obj_val;
                result = r;

                // loop over the rows of the optimal sequence
                for (size_t i = 1; i < ph() + 1; i++)
                {
                    for (size_t j = 0; j < nx(); j++)
                    {
                        sequence.state.row(i - 1)[j] = work->solution->x[i * (nx() + nu()) + j];
                    }

                    for (size_t j = 0; j < nu(); j++)
                    {
                        sequence.input.row(i - 1)[j] = work->solution->x[((ph() + 1) * (nx() + nu())) + (i * nu()) + j];
                    }

                    sequence.output.row(i - 1) = builder->mapToOutput(sequence.state.row(i - 1), extInputMeas.col(i - 1));
                }
            }
            else
            {
                r.cost = mpc::inf;
                r.cmd = result.cmd;
                r.retcode = -1;

                sequence.state.setZero();
                sequence.input.setZero();
                sequence.output.setZero();
            }

            clearData();
        }

    private:
        /**
         * @brief Create an osqp sparse matrix from a sparse eigen matrix
         *
         * @param eigenSparseMatrix sparse eigen matrix
         * @param osqpSparseMatrix osqp sparse matrix
         * @return true
         * @return false
         */
        bool createOsqpSparseMatrix(const smat &eigenSparseMatrix, csc *&osqpSparseMatrix)
        {
            // Copying into a new sparse matrix to be sure to use a CSC matrix
            // this may perform memory allocation, but this is already the case
            // for allocating the osqpSparseMatrix
            smat colMajorCopy = eigenSparseMatrix;

            // get number of row, columns and nonZeros from Eigen SparseMatrix
            c_int rows = colMajorCopy.rows();
            c_int cols = colMajorCopy.cols();
            c_int numberOfNonZeroCoeff = colMajorCopy.nonZeros();

            // get innerr and outer index
            const int *outerIndexPtr = colMajorCopy.outerIndexPtr();
            const int *innerNonZerosPtr = colMajorCopy.innerNonZeroPtr();

            if (osqpSparseMatrix != nullptr)
            {
                return false;
            }

            // instantiate csc matrix
            osqpSparseMatrix = csc_spalloc(rows, cols, numberOfNonZeroCoeff, 1, 0);

            int innerOsqpPosition = 0;
            for (int k = 0; k < cols; k++)
            {
                if (colMajorCopy.isCompressed())
                {
                    osqpSparseMatrix->p[k] = static_cast<c_int>(outerIndexPtr[k]);
                }
                else
                {
                    if (k == 0)
                    {
                        osqpSparseMatrix->p[k] = 0;
                    }
                    else
                    {
                        osqpSparseMatrix->p[k] = osqpSparseMatrix->p[k - 1] + innerNonZerosPtr[k - 1];
                    }
                }
                for (typename smat::InnerIterator it(colMajorCopy, k); it; ++it)
                {
                    osqpSparseMatrix->i[innerOsqpPosition] = static_cast<c_int>(it.row());
                    osqpSparseMatrix->x[innerOsqpPosition] = static_cast<c_float>(it.value());
                    innerOsqpPosition++;
                }
            }

            osqpSparseMatrix->p[static_cast<int>(cols)] = static_cast<c_int>(innerOsqpPosition);
            assert(innerOsqpPosition == numberOfNonZeroCoeff);
            return true;
        }

        /**
         * @brief Clear the current allocated osqp problem data structures
         */
        void clearData()
        {
            osqp_cleanup(work);

            if (data)
            {
                if (data->A)
                {
                    csc_spfree(data->A);
                }
                if (data->P)
                {
                    csc_spfree(data->P);
                }
                c_free(data);
            }

            if (settings)
            {
                c_free(settings);
            }
        }

        /**
         * @brief Initialize the osqp problem data structures
         */
        void initData()
        {
            settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
            data = (OSQPData *)c_malloc(sizeof(OSQPData));

            data->P = nullptr;
            data->A = nullptr;
        }

        OSQPWorkspace *work;
        OSQPSettings *settings;
        OSQPData *data;
        c_int exitflag = 0;

        mat<sizer.ny, sizer.ph> outSysRef;
        mat<sizer.nu, sizer.ph> cmdSysRef, deltaCmdSysRef;
        mat<sizer.ndu, sizer.ph> extInputMeas;

        ProblemBuilder<sizer> *builder;
    };
} // namespace mpc
