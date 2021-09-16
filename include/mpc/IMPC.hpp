#pragma once

#include <mpc/Common.hpp>
#include <mpc/IOptimizer.hpp>

#include <chrono>

namespace mpc {
template <
    int Tnx, int Tnu, int Tndu, int Tny,
    int Tph, int Tch, int Tineq, int Teq>
class IMPC : public Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq> {

public:
    virtual bool setContinuosTimeModel(const double) = 0;
    virtual void setInputScale(const cvec<Tnu>) = 0;
    virtual void setStateScale(const cvec<Tnx>) = 0;
    virtual void setOptimizerParameters(const Parameters) = 0;

    void onInit()
    {
        onSetup();

        result.cmd.resize(dim.nu.num());
        result.cmd.setZero();
    };

    bool setLoggerLevel(Logger::log_level l)
    {
        Logger::instance().setLevel(l);
        return true;
    }

    bool setLoggerPrefix(std::string prefix)
    {
        Logger::instance().setPrefix(prefix);
        return true;
    }

    Result<Tnu> step(const cvec<Tnx> x0, const cvec<Tnu> lastU)
    {
        checkOrQuit();

        onModelUpdate(x0);

        Logger::instance().log(Logger::log_type::INFO)
            << "Optimization step"
            << std::endl;

        auto start = std::chrono::steady_clock::now();
        result = optPtr->run(x0, lastU);
        auto stop = std::chrono::steady_clock::now();

        Logger::instance().log(Logger::log_type::INFO)
            << "Optimization step duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
            << " (ms)"
            << std::endl;
        return result;
    }

    Result<Tnu> getLastResult()
    {
        checkOrQuit();
        return result;
    }

protected:
    virtual void onSetup() = 0;
    virtual void onModelUpdate(const cvec<Tnx>) = 0;

    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>::checkOrQuit;
    using Common<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>::dim;

    IOptimizer<Tnx, Tnu, Tndu, Tny, Tph, Tch, Tineq, Teq>* optPtr;
    Result<Tnu> result;
};
}