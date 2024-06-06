#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include <functional>
#include <mpc/LMPC.hpp>
#include <mpc/NLMPC.hpp>

namespace py = pybind11;

template <int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq, int Teq>
void expose_NLMPC(py::module &m)
{
    std::string name = "NLMPC";
    using NLMPCType = mpc::NLMPC<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>;

    using StateSpaceFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &, const unsigned int &)>;
    auto stateSpaceFuncWrapper = [](NLMPCType &self, StateSpaceFunc impl, double tol)
    {
        return self.setStateSpaceFunction([impl, tol](Eigen::VectorXd &xd, const Eigen::VectorXd &x, const Eigen::VectorXd &u, const unsigned int &k)
        { 
            // invoke the python function and assign the result to reference value
            xd = impl(x, u, k); 
        }, tol);
    };

    using OutputFunc = StateSpaceFunc;
    auto outputFuncWrapper = [](NLMPCType &self, OutputFunc impl)
    {
        return self.setOutputFunction([impl](Eigen::VectorXd &y, const Eigen::VectorXd &x, const Eigen::VectorXd &u, const unsigned int &k)
        { 
            // invoke the python function and assign the result to reference value
            y = impl(x, u, k); 
        });
    };

    using IneqConFunc = std::function<Eigen::VectorXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const double &)>;
    auto ineqConFuncWrapper = [](NLMPCType &self, IneqConFunc impl, double tol)
    {
        return self.setIneqConFunction([impl, tol](Eigen::VectorXd &cineq, const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, const Eigen::MatrixXd &u, const double &slack)
        { 
            // invoke the python function and assign the result to reference value
            cineq = impl(x, y, u, slack);
        }, tol);
    };

    using EqConFunc = std::function<Eigen::VectorXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &)>;
    auto eqConFuncWrapper = [](NLMPCType &self, EqConFunc impl, double tol)
    {
        return self.setEqConFunction([impl, tol](Eigen::VectorXd &ceq, const Eigen::MatrixXd &x, const Eigen::MatrixXd &u)
        {
            // invoke the python function and assign the result to reference value
            ceq = impl(x, u);
        }, tol);
    };

    py::class_<NLMPCType>(m, name.c_str())
        .def(py::init<const int &, const int &, const int &, const int &, const int &, const int &, const int &>())
        // methods from the IMPC class
        .def("setDiscretizationSamplingTime", &NLMPCType::setDiscretizationSamplingTime)
        .def("setInputScale", &NLMPCType::setInputScale)
        .def("setStateScale", &NLMPCType::setStateScale)
        .def("setOptimizerParameters", &NLMPCType::setOptimizerParameters)
        .def("setLoggerLevel", &NLMPCType::setLoggerLevel)
        .def("setLoggerPrefix", &NLMPCType::setLoggerPrefix)
        .def("optimize", &NLMPCType::optimize)
        .def("getLastResult", &NLMPCType::getLastResult)
        .def("getOptimalSequence", &NLMPCType::getOptimalSequence)
        .def("getExecutionStats", &NLMPCType::getExecutionStats)
        .def("resetStats", &NLMPCType::resetStats)
        .def("setStateBounds", py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&NLMPCType::setStateBounds))
        .def("setStateBounds", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &, const mpc::HorizonSlice &>(&NLMPCType::setStateBounds))
        .def("setInputBounds", py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&NLMPCType::setInputBounds))
        .def("setInputBounds", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &, const mpc::HorizonSlice &>(&NLMPCType::setInputBounds))
        .def("setOutputBounds", py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&NLMPCType::setOutputBounds))
        .def("setOutputBounds", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &, const mpc::HorizonSlice &>(&NLMPCType::setOutputBounds))
        // methods from the NLMPC class
        .def("setObjectiveFunction", &NLMPCType::setObjectiveFunction)
        .def("setStateSpaceFunction", stateSpaceFuncWrapper)
        .def("setOutputFunction", outputFuncWrapper)
        .def("setIneqConFunction", ineqConFuncWrapper)
        .def("setEqConFunction", eqConFuncWrapper);
}

template <int Tnx, int Tnu, int Tndu, int Tny, int Tph, int Tch>
void expose_LMPC(py::module &m)
{
    std::string name = "LMPC";
    using LMPCType = mpc::LMPC<Tnx, Tnu, Tndu, Tny, Tph, Tch>;

    py::class_<LMPCType>(m, name.c_str())
        .def(py::init<const int &, const int &, const int &, const int &, const int &, const int &>())
        // methods from the IMPC class
        .def("setOptimizerParameters", &LMPCType::setOptimizerParameters)
        .def("setLoggerLevel", &LMPCType::setLoggerLevel)
        .def("setLoggerPrefix", &LMPCType::setLoggerPrefix)
        .def("optimize", &LMPCType::optimize)
        .def("getLastResult", &LMPCType::getLastResult)
        .def("getOptimalSequence", &LMPCType::getOptimalSequence)
        .def("getExecutionStats", &LMPCType::getExecutionStats)
        .def("resetStats", &LMPCType::resetStats)
        .def("setStateBounds", py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&LMPCType::setStateBounds))
        .def("setStateBounds", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &, const mpc::HorizonSlice &>(&LMPCType::setStateBounds))
        .def("setInputBounds", py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&LMPCType::setInputBounds))
        .def("setInputBounds", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &, const mpc::HorizonSlice &>(&LMPCType::setInputBounds))
        .def("setOutputBounds", py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&LMPCType::setOutputBounds))
        .def("setOutputBounds", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &, const mpc::HorizonSlice &>(&LMPCType::setOutputBounds))
        // methods from the LMPC class
        .def("setStateSpaceModel", &LMPCType::setStateSpaceModel)
        .def("setDisturbances", &LMPCType::setDisturbances)
        .def("getSolverWarmStartPrimal", &LMPCType::getSolverWarmStartPrimal)
        .def("getSolverWarmStartDual", &LMPCType::getSolverWarmStartDual)
        .def("setSolverWarmStart", &LMPCType::setSolverWarmStart)
        .def("setObjectiveWeights", py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&LMPCType::setObjectiveWeights))
        .def("setObjectiveWeights", py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &, const mpc::HorizonSlice&>(&LMPCType::setObjectiveWeights))
        .def("setScalarConstraint", py::overload_cast<const unsigned int, const double, const double, const Eigen::VectorXd, const Eigen::VectorXd>(&LMPCType::setScalarConstraint))
        .def("setScalarConstraint", py::overload_cast<const double, const double, const Eigen::VectorXd, const Eigen::VectorXd, const mpc::HorizonSlice&>(&LMPCType::setScalarConstraint))
        .def("setExogenousInputs", py::overload_cast<const Eigen::MatrixXd &>(&LMPCType::setExogenousInputs))
        .def("setExogenousInputs", py::overload_cast<const Eigen::VectorXd &, const mpc::HorizonSlice&>(&LMPCType::setExogenousInputs))
        .def("setReferences", py::overload_cast<const Eigen::VectorXd, const Eigen::VectorXd, const Eigen::VectorXd, const mpc::HorizonSlice&>(&LMPCType::setReferences))
        .def("setReferences", py::overload_cast<const Eigen::VectorXd, const Eigen::VectorXd, const Eigen::VectorXd, const mpc::HorizonSlice&>(&LMPCType::setReferences));
}

PYBIND11_MODULE(pympcxx, m)
{
    // export the API for the NLMPC class
    expose_NLMPC<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic>(m);

    // export the API for the LMPC class
    expose_LMPC<Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic>(m);

    // export the logger level enum
    py::enum_<mpc::Logger::log_level>(m, "LoggerLevel")
        .value("DEEP", mpc::Logger::log_level::DEEP)
        .value("NORMAL", mpc::Logger::log_level::NORMAL)
        .value("ALERT", mpc::Logger::log_level::ALERT)
        .value("NONE", mpc::Logger::log_level::NONE)
        .export_values();

    // export the result struct to python
    using ResulType = mpc::Result<Eigen::Dynamic>;

    py::class_<ResulType>(m, "Result")
        .def_readonly("solver_status", &ResulType::solver_status)
        .def_readonly("solver_status_msg", &ResulType::solver_status_msg)
        .def_readonly("cost", &ResulType::cost)
        .def_readonly("cmd", &ResulType::cmd)
        .def_readonly("status", &ResulType::status);

    // export the solution stats struct to python
    using StatsType = mpc::SolutionStats;

    py::class_<StatsType>(m, "SolutionStats")
        .def_readonly("totalSolutionTime", &StatsType::totalSolutionTime)
        .def_readonly("numberOfSolutions", &StatsType::numberOfSolutions)
        .def_readonly("minSolutionTime", &StatsType::minSolutionTime)
        .def_readonly("maxSolutionTime", &StatsType::maxSolutionTime)
        .def_readonly("averageSolutionTime", &StatsType::averageSolutionTime)
        .def_readonly("standardDeviation", &StatsType::standardDeviation)
        .def_readonly("solutionsStates", &StatsType::solutionsStates);

    // export the status enum
    py::enum_<mpc::ResultStatus>(m, "ResultStatus")
        .value("UNKNOWN", mpc::ResultStatus::UNKNOWN)
        .value("SUCCESS", mpc::ResultStatus::SUCCESS)
        .value("MAX_ITERATION", mpc::ResultStatus::MAX_ITERATION)
        .value("INFEASIBLE", mpc::ResultStatus::INFEASIBLE)
        .value("ERROR", mpc::ResultStatus::ERROR)
        .export_values();

    // export HorizonSlice struct to python
    py::class_<mpc::HorizonSlice>(m, "HorizonSlice")
        .def(py::init<int, int>())
        .def_static("all", &mpc::HorizonSlice::all);
}