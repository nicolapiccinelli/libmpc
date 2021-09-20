#pragma once

#include <mpc/IOptimizer.hpp>
#include <mpc/ProblemBuilder.hpp>

#include <osqp/osqp.h>

namespace mpc {
/**
 * @brief Linear MPC optimizer interface class
 * 
 * @tparam Tnx dimension of the state space
 * @tparam Tnu dimension of the input space
 * @tparam Tndu dimension of the measured disturbance space
 * @tparam Tny dimension of the output space
 * @tparam Tph length of the prediction horizon
 * @tparam Tch length of the control horizon
 */
template <
    int Tnx, int Tnu, int Tndu,
    int Tny, int Tph, int Tch>
class LOptimizer : public IOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0> {
private:
    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::checkOrQuit;
    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, 0, 0>::dim;

public:
    LOptimizer() = default;

    ~LOptimizer()
    {
        checkOrQuit();

        clearData();

        if (settings) {
            c_free(settings);
        }

        if (work) {
            c_free(work);
        }
    }

    /**
     * @brief Initialization hook override. Performing initialization in this
     * method ensures the correct problem dimensions assigment has been
     * already performed
     */
    void onInit()
    {
        settings = (OSQPSettings*)c_malloc(sizeof(OSQPSettings));

        initData();

        setParameters(LParameters());

        last_r.cmd.resize(dim.nu.num());
        last_r.cmd.setZero();

        extInputMeas.resize(dim.ndu.num());
        outSysRef.resize(dim.ny.num());
        cmdSysRef.resize(dim.nu.num());
        deltaCmdSysRef.resize(dim.nu.num());

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
    void setBuilder(ProblemBuilder<Tnx, Tnu, Tndu, Tny, Tph, Tch>* b)
    {
        checkOrQuit();
        builder = b;
    }

    /**
     * @brief Set the optmiziation parameters
     * 
     * @param param parameters desired
     */
    void setParameters(const Parameters& param)
    {
        checkOrQuit();

        auto lin_params = dynamic_cast<const LParameters*>(&param);

        // define solver settings as default
        if (settings) {
            osqp_set_default_settings(settings);
            
            settings->alpha = lin_params->alpha;
            settings->verbose = lin_params->verbose;
            settings->rho = lin_params->rho;
            settings->adaptive_rho = lin_params->adaptive_rho;
            settings->eps_rel = lin_params->eps_rel;
            settings->eps_abs = lin_params->eps_abs;
            settings->eps_prim_inf = lin_params->eps_prim_inf;
            settings->eps_dual_inf = lin_params->eps_dual_inf;
            settings->max_iter = lin_params->maximum_iteration;
            settings->polish = lin_params->polish;
        }

        Logger::instance().log(Logger::log_type::DETAIL)
            << "Setting tolerances and stopping criterias"
            << std::endl;
    }

    /**
     * @brief Set the references vector for the objective function
     * 
     * @param outRef reference for the output
     * @param cmdRef reference for the optimal control input
     * @param deltaCmdRef reference for the variation of the optimal control input
     * @return true 
     * @return false 
     */
    bool setReferences(
        const cvec<Tny>& outRef,
        const cvec<Tnu>& cmdRef,
        const cvec<Tnu>& deltaCmdRef)
    {
        outSysRef = outRef;
        cmdSysRef = cmdRef;
        deltaCmdSysRef = deltaCmdRef;

        return true;
    }

    /**
     * @brief Set the exogenuos inputs vector
     * 
     * @param uMeas measured exogenuos input
     * @return true 
     * @return false 
     */
    bool setExogenuosInputs(const cvec<Tndu>& uMeas)
    {
        extInputMeas = uMeas;
        return true;
    }

    /**
     * @brief Implementation of the optimization step
     * 
     * @param x0 system's variables initial condition
     * @param u0 control action initial condition for warm start
     * @return Result<Tnu> optimization result
     */
    Result<Tnu> run(
        const cvec<Tnx>& x0,
        const cvec<Tnu>& u0)
    {
        checkOrQuit();
        Result<Tnu> r;

        auto mpcProblem = builder->get(x0, u0, outSysRef, cmdSysRef, deltaCmdSysRef, extInputMeas);
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
        clearData();
        initData();

        if (data) {
            data->n = numVars;
            data->m = numConstraints;

            if (!createOsqpSparseMatrix(P, data->P)) {
                Logger::instance().log(Logger::log_type::ERROR) << "Unable to create the P matrix" << std::endl;
            }

            data->q = mpcProblem.q.data();

            if (!createOsqpSparseMatrix(A, data->A)) {
                Logger::instance().log(Logger::log_type::ERROR) << "Unable to create the A matrix" << std::endl;
            }

            data->l = mpcProblem.l.data();
            data->u = mpcProblem.u.data();
        }

        // setup workspace
        exitflag = osqp_setup(&work, data, settings);
        if (exitflag > 0) {
            Logger::instance().log(Logger::log_type::ERROR) << "Unable to solve " << exitflag << std::endl;
        }

        // solve problem
        osqp_solve(work);

        // if the solution is valid update the
        if (work->solution->x != NULL) {
            int index = 0;

            cvec<dim.nu> optCmd;
            optCmd.resize(dim.nu.num());

            for (size_t i = ((dim.ph.num() + 1) * (dim.nx.num() + dim.nu.num())); i < (((dim.ph.num() + 1) * (dim.nx.num() + dim.nu.num())) + dim.nu.num()); i++) {
                optCmd[index++] = work->solution->x[i];
            }

            r.cmd = optCmd;
            r.retcode = work->info->status_val;
            r.cost = work->info->obj_val;
            last_r = r;
        } else {
            r = last_r;
        }

        return r;
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
    bool createOsqpSparseMatrix(const smat& eigenSparseMatrix, csc*& osqpSparseMatrix)
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
        const int* outerIndexPtr = colMajorCopy.outerIndexPtr();
        const int* innerNonZerosPtr = colMajorCopy.innerNonZeroPtr();

        if (osqpSparseMatrix != nullptr) {
            return false;
        }

        // instantiate csc matrix
        osqpSparseMatrix = csc_spalloc(rows, cols, numberOfNonZeroCoeff, 1, 0);

        int innerOsqpPosition = 0;
        for (int k = 0; k < cols; k++) {
            if (colMajorCopy.isCompressed()) {
                osqpSparseMatrix->p[k] = static_cast<c_int>(outerIndexPtr[k]);
            } else {
                if (k == 0) {
                    osqpSparseMatrix->p[k] = 0;
                } else {
                    osqpSparseMatrix->p[k] = osqpSparseMatrix->p[k - 1] + innerNonZerosPtr[k - 1];
                }
            }
            for (typename smat::InnerIterator it(colMajorCopy, k); it; ++it) {
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
        if (data) {
            if (data->A) {
                c_free(data->A);
            }
            if (data->P) {
                c_free(data->P);
            }
            c_free(data);
        }
    }

    /**
     * @brief Initialize the osqp problem data structures
     */
    void initData()
    {
        data = (OSQPData*)c_malloc(sizeof(OSQPData));
        data->P = nullptr;
        data->A = nullptr;
    }

    OSQPWorkspace* work;
    OSQPSettings* settings;
    OSQPData* data;
    c_int exitflag = 0;

    Result<Tnu> last_r;
    double currentSlack;
    bool hard;

    cvec<Tny> outSysRef;
    cvec<Tnu> cmdSysRef, deltaCmdSysRef;
    cvec<Tndu> extInputMeas;

    ProblemBuilder<Tnx, Tnu, Tndu, Tny, Tph, Tch>* builder;
};
} // namespace mpc
