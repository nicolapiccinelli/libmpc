#pragma once

#include <mpc/types.hpp>

namespace mpc
{
    inline constexpr int getDim(const int n, bool c)
    {
        if (c)
        {
            return n;
        }
        else
        {
            return Eigen::Dynamic;
        }
    }

    struct MPCDims
    {
        int tnx = 0;
        int tnu = 0;
        int tny = 0;
        int tph = 0;
        int tch = 0;
        int tineq = 0;
        int teq = 0;
    };
    
    enum sizeEnum
    {
        StateEqSize,
        StateIneqSize,
        TphPlusOne,
        InputEqSize,
        InputPredictionSize,
        DecVarsSize,
        StateIneqCostSize,
        UserIneqCostSize,
        StateEqCostSize,
        UserEqCostSize
    };

    template <
        int Tnx, int Tnu, int Tny,
        int Tph, int Tch,
        int Tineq, int Teq>
    class Common
    {
    public:
        Common()
        {
            _isInitialized = false;
        }

        inline int GetSize(sizeEnum size)
        {
            switch (size)
            {
            case sizeEnum::StateEqSize:
                return _dimensions.tnx * _dimensions.tph;
                break;
            case sizeEnum::StateIneqSize:
                return 2 * _dimensions.tph * _dimensions.tny;
                break;
            case sizeEnum::TphPlusOne:
                return _dimensions.tph + 1;
                break;
            case sizeEnum::InputEqSize:
                return _dimensions.tnu * _dimensions.tch;
                break;
            case sizeEnum::InputPredictionSize:
                return _dimensions.tph * _dimensions.tnu;
                break;
            case sizeEnum::DecVarsSize:
                return GetSize(sizeEnum::StateEqSize) + GetSize(sizeEnum::InputEqSize) + 1;
                break;
            case sizeEnum::StateIneqCostSize:
                return GetSize(sizeEnum::StateIneqSize) * GetSize(sizeEnum::DecVarsSize);
                break;
            case sizeEnum::UserIneqCostSize:
                return _dimensions.tineq * GetSize(sizeEnum::DecVarsSize);
                break;
            case sizeEnum::StateEqCostSize:
                return GetSize(StateEqSize) * GetSize(sizeEnum::DecVarsSize);
                break;
            case sizeEnum::UserEqCostSize:
                return _dimensions.teq * GetSize(sizeEnum::DecVarsSize);
                break;
            default:
                return 0;
                break;
            }            
        }

        static inline constexpr int MultiplySize(int sizeA, int sizeB)
        {
            if(sizeA > Eigen::Dynamic && sizeB > Eigen::Dynamic)
            {
                return sizeA * sizeB;
            }
            else
            {
                return Eigen::Dynamic;
            }
        }

        static inline constexpr int AssignSize(sizeEnum size)
        {
            switch (size)
            {
            case sizeEnum::StateEqSize:
                return getDim(
                    Tnx * Tph,
                    Tnx > Eigen::Dynamic && Tph > Eigen::Dynamic);
                break;
            case sizeEnum::StateIneqSize:
                return getDim(
                    2 * Tph * Tny,
                    Tph > Eigen::Dynamic && Tny > Eigen::Dynamic);
                break;
            case sizeEnum::TphPlusOne:
                return getDim(
                    Tph + 1,
                    Tph > Eigen::Dynamic);
                break;
            case sizeEnum::InputEqSize:
                return getDim(
                    Tnu * Tch,
                    Tnu > Eigen::Dynamic && Tch > Eigen::Dynamic);
                break;
            case sizeEnum::InputPredictionSize:
                return getDim(
                    Tph * Tnu,
                    Tph > Eigen::Dynamic && Tnu > Eigen::Dynamic);
                break;
            case sizeEnum::DecVarsSize:
                return getDim(
                    AssignSize(sizeEnum::StateEqSize) + AssignSize(sizeEnum::InputEqSize) + 1,
                    AssignSize(sizeEnum::StateEqSize) > Eigen::Dynamic && AssignSize(sizeEnum::InputEqSize) > Eigen::Dynamic);
                break;
            case sizeEnum::StateIneqCostSize:
                return getDim(
                    AssignSize(sizeEnum::StateIneqSize) * AssignSize(sizeEnum::DecVarsSize),
                    AssignSize(sizeEnum::StateIneqSize) > Eigen::Dynamic && AssignSize(sizeEnum::DecVarsSize) > Eigen::Dynamic);
                break;
            case sizeEnum::UserIneqCostSize:
                return getDim(
                    Tineq * AssignSize(sizeEnum::DecVarsSize),
                    Tineq > Eigen::Dynamic && AssignSize(sizeEnum::DecVarsSize) > Eigen::Dynamic);
                break;
            case sizeEnum::StateEqCostSize:
                return getDim(
                    AssignSize(sizeEnum::StateEqSize) * AssignSize(sizeEnum::DecVarsSize),
                    AssignSize(sizeEnum::StateEqSize) > Eigen::Dynamic && AssignSize(sizeEnum::DecVarsSize) > Eigen::Dynamic);
                break;
            case sizeEnum::UserEqCostSize:
                return getDim(
                    Teq * AssignSize(sizeEnum::DecVarsSize),
                    Teq > Eigen::Dynamic && AssignSize(sizeEnum::DecVarsSize) > Eigen::Dynamic);
                break;
            default:
                return 0;
                break;
            }
        }

        using ObjFunHandle = std::function<double(
            mat<AssignSize(sizeEnum::TphPlusOne), Tnx>, 
            mat<AssignSize(sizeEnum::TphPlusOne), Tnu>, 
            double)>;
        using IConFunHandle = std::function<void(
            cvec<Tineq> &, 
            mat<AssignSize(sizeEnum::TphPlusOne), Tnx>, 
            mat<AssignSize(sizeEnum::TphPlusOne), Tnu>, 
            double)>;
        using EConFunHandle = std::function<void(
            cvec<Teq> &, 
            mat<AssignSize(sizeEnum::TphPlusOne), Tnx>, 
            mat<AssignSize(sizeEnum::TphPlusOne), Tnu>)>;
        using StateFunHandle = std::function<void(
            cvec<Tnx> &, 
            cvec<Tnx>, 
            cvec<Tnu>)>;
        using OutFunHandle = std::function<void(void)>;

        struct Result
        {
            Result() :
                retcode(0),
                cost(0)
            {
                cmd.setZero();
            }

            int retcode;
            double cost;
            cvec<Tnu> cmd;
        };

    protected:
        inline void _initialize(
            int tnx, int tnu, int tny,
            int tph, int tch,
            int tineq, int teq)
        {
            _dimensions.tnx = tnx;
            _dimensions.tnu = tnu;
            _dimensions.tny = tny;
            _dimensions.tph = tph;
            _dimensions.tch = tch;
            _dimensions.tineq = tineq;
            _dimensions.teq = teq;

            _isInitialized = true;
        }

        inline void _checkOrQuit()
        {
            if (!_isInitialized)
            {
                Logger::instance().log(Logger::log_type::ERROR) << RED << "MPC library is not initialized, quitting..." << RESET << std::endl;
                exit(-1);
            }
        }

        MPCDims _dimensions;
        
    private:
        bool _isInitialized;
    };

} // namespace mpc
