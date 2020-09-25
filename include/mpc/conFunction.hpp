#pragma once

#include <mpc/baseFunction.hpp>

namespace mpc
{
    template <
        int Tnx, int Tnu, int Tny,
        int Tph, int Tch,
        int Tineq, int Teq>
    class ConFunction : public BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>
    {
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_mapping;
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_x0;
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_Xmat;
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_Umat;
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_e;
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_ts;
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_ctime;
        using BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_niteration;

        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_initialize;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_checkOrQuit;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::GetSize;
        using Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::_dimensions;

    public:
        template <int Tcon=Eigen::Dynamic>
        struct Cost
        {
            cvec<Tcon> value;
            cvec<Tcon * AssignSize(sizeEnum::DecVarsSize)> grad;
        };

        // template <>
        // struct Cost<Eigen::Dynamic>
        // {
        //     cvec<Eigen::Dynamic> value;
        //     cvec<Eigen::Dynamic> grad;
        // };

        ConFunction() : BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>() {}
        ~ConFunction() = default;

        void initialize(
            int tnx, int tnu, int tny,
            int tph, int tch,
            int tineq, int teq)
        {
            _initialize(tnx, tnu, tny, tph, tch, tineq, teq);

            _nx = 0;

            _x0.resize(_dimensions.tnx);
            _Xmat.resize(GetSize(sizeEnum::TphPlusOne), _dimensions.tnx);
            _Umat.resize(GetSize(sizeEnum::TphPlusOne), _dimensions.tnu);

            _ceq.resize(GetSize(sizeEnum::StateEqSize));
            _Jceq.resize(GetSize(sizeEnum::DecVarsSize), GetSize(sizeEnum::StateEqSize));

            _cineq.resize(GetSize(sizeEnum::StateIneqSize));
            _Jcineq.resize(GetSize(sizeEnum::DecVarsSize), GetSize(sizeEnum::StateIneqSize));

            _ceq_user.resize(_dimensions.teq);
            _Jceq_user.resize(GetSize(sizeEnum::DecVarsSize), _dimensions.teq);

            _cineq_user.resize(_dimensions.tineq);
            _Jcineq_user.resize(GetSize(sizeEnum::DecVarsSize), _dimensions.tineq);
        }

        bool hasIneqConstraintFunction()
        {
            _checkOrQuit();
            return _ieqUser != nullptr;
        }

        bool hasEqConstraintFunction()
        {
            _checkOrQuit();
            return _eqUser != nullptr;
        }

        bool setStateSpaceFunction(
            const typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::StateFunHandle handle)
        {
            _checkOrQuit();
            return _fUser = handle, true;
        }

        bool setOutputFunction(
            const typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::OutFunHandle handle)
        {
            _checkOrQuit();
            return _outUser = handle, true;
        }

        bool setIneqConstraintFunction(
            const typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle handle)
        {
            _checkOrQuit();
            return _ieqUser = handle, true;
        }

        bool setEqConstraintFunction(
            const typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle handle)
        {
            _checkOrQuit();
            return _eqUser = handle, true;
        }

        Cost<AssignSize(sizeEnum::StateIneqSize)> evaluateIneq(
            cvec<AssignSize(sizeEnum::DecVarsSize)> x,
            bool hasGradient)
        {
            _checkOrQuit();

            _mapping.unwrapVector(x, _x0, _Xmat, _Umat, _e);

            // Set MPC constraints
            _getStateIneqConstraints();

            static Cost<AssignSize(sizeEnum::StateIneqSize)> c;
            c.value = _cineq;
            c.grad = Eigen::Map<cvec<AssignSize(sizeEnum::StateIneqCostSize)>>(
                _Jcineq.data(),
                _Jcineq.size());

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "State inequality constraints value:\n"
                << std::setprecision(10) 
                << c.value 
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "Gradient state inequality constraints not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "State inequality constraints gradient:\n"
                    << std::setprecision(10) 
                    << c.grad 
                    << std::endl;
            }

            return c;
        }

        Cost<Tineq> evaluateUserIneq(
            cvec<AssignSize(sizeEnum::DecVarsSize)> x,
            bool hasGradient)
        {
            _checkOrQuit();

            _mapping.unwrapVector(x, _x0, _Xmat, _Umat, _e);

            // Add user defined constraints
            if (hasIneqConstraintFunction())
            {
                _ieqUser(_cineq_user, _Xmat, _Umat, _e);

                static mat<Tineq, AssignSize(sizeEnum::StateEqSize)> Jieqx;
                Jieqx.resize(_dimensions.tineq, GetSize(sizeEnum::StateEqSize));

                static mat<Tineq, AssignSize(sizeEnum::InputPredictionSize)> Jieqmv;
                Jieqmv.resize(_dimensions.tineq, GetSize(sizeEnum::InputPredictionSize));

                static cvec<Tineq> Jie;
                Jie.resize(_dimensions.tineq);

                _computeUserIneqJacobian(
                    Jieqx,
                    Jieqmv,
                    Jie,
                    _Xmat,
                    _Umat,
                    _e,
                    _cineq_user);

                _glueJacobian<Tineq>(
                    _Jcineq_user,
                    Jieqx,
                    Jieqmv,
                    Jie);

                // TODO support for jacobian scaling
            }
            else
            {
                _cineq_user.setZero();
                _Jcineq_user.setZero();
            }

            static Cost<Tineq> c;
            c.value = _cineq_user;
            c.grad = Eigen::Map<cvec<AssignSize(sizeEnum::UserIneqCostSize)>>(
                _Jcineq_user.data(),
                _Jcineq_user.size());

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "User inequality constraints value:\n"
                << std::setprecision(10) 
                << c.value 
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                << "Gradient user inequality constraints not currently used"
                << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "User inequality constraints gradient:\n"
                    << std::setprecision(10) 
                    << c.grad 
                    << std::endl;
            }

            return c;
        }

        Cost<AssignSize(sizeEnum::StateEqSize)> evaluateEq(
            cvec<AssignSize(sizeEnum::DecVarsSize)> x,
            bool hasGradient)
        {
            _checkOrQuit();
            _mapping.unwrapVector(x, _x0, _Xmat, _Umat, _e);

            // Set MPC constraints
            _getStateEqConstraints(hasGradient);

            static Cost<AssignSize(sizeEnum::StateEqSize)> c;
            c.value = _ceq;
            c.grad = Eigen::Map<cvec<AssignSize(sizeEnum::StateEqCostSize)>>(
                _Jceq.data(),
                _Jceq.size());

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "State equality constraints value:\n"
                << std::setprecision(10) 
                << c.value 
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "State equality constraints gradient not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "State equality constraints gradient:\n"
                    << std::setprecision(10) 
                    << c.grad 
                    << std::endl;
            }

            return c;
        }

        Cost<Teq> evaluateUserEq(
            cvec<AssignSize(sizeEnum::DecVarsSize)> x,
            bool hasGradient)
        {
            _checkOrQuit();
            _mapping.unwrapVector(x, _x0, _Xmat, _Umat, _e);

            // Add user defined constraints
            if (hasEqConstraintFunction())
            {
                _eqUser(_ceq_user, _Xmat, _Umat);

                static mat<Teq, AssignSize(sizeEnum::StateEqSize)> Jeqx;
                Jeqx.resize(_dimensions.teq, GetSize(sizeEnum::StateEqSize));

                static mat<Teq, AssignSize(sizeEnum::InputPredictionSize)> Jeqmv;
                Jeqmv.resize(_dimensions.teq, GetSize(sizeEnum::InputPredictionSize));

                _computeUserEqJacobian(
                    Jeqx,
                    Jeqmv,
                    _Xmat,
                    _Umat,
                    _ceq_user);

                _glueJacobian<Teq>(
                    _Jceq_user,
                    Jeqx,
                    Jeqmv,
                    cvec<Teq>::Zero(_dimensions.teq));

                // TODO support for jacobian scaling
            }
            else
            {
                _ceq_user.setZero();
                _Jceq_user.setZero();
            }

            static Cost<Teq> c;
            c.value = _ceq_user;
            c.grad = Eigen::Map<cvec<mpc::Common<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::AssignSize(sizeEnum::UserEqCostSize)>>(
                _Jceq_user.data(),
                _Jceq_user.size());

            Logger::instance().log(Logger::log_type::DEBUG) 
                << "User equality constraints value:\n"
                << std::setprecision(10) 
                << c.value 
                << std::endl;
            if (!hasGradient)
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "Gradient user equality constraints not currectly used"
                    << std::endl;
            }
            else
            {
                Logger::instance().log(Logger::log_type::DEBUG) 
                    << "User equality constraints gradient:\n"
                    << std::setprecision(10) 
                    << c.grad 
                    << std::endl;
            }

            return c;
        }

    private:
        template <int Tnc>
        void
        _glueJacobian(
            mat<AssignSize(sizeEnum::DecVarsSize), Tnc> &Jres,
            mat<Tnc, AssignSize(sizeEnum::StateEqSize)> Jstate,
            mat<Tnc, AssignSize(sizeEnum::InputPredictionSize)> Jmanvar,
            cvec<Tnc> Jcon)
        {
            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tph; i++)
            {
                Jres.middleRows(i * _dimensions.tnx, _dimensions.tnx) =
                    Jstate.middleCols(i * _dimensions.tnx, _dimensions.tnx).transpose();
            }

            static mat<Tnc, AssignSize(sizeEnum::InputPredictionSize)> Jmanvar_mat;
            Jmanvar_mat.resize(Jres.cols(), GetSize(sizeEnum::InputPredictionSize));

            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tph; i++)
            {
                Jmanvar_mat.block(0, i * _dimensions.tnu, Jres.cols(), _dimensions.tnu) =
                    Jmanvar.middleCols(i * _dimensions.tnu, _dimensions.tnu);
            }

            Jres.middleRows(GetSize(sizeEnum::StateEqSize), GetSize(sizeEnum::InputEqSize)) =
                (Jmanvar_mat * _mapping.Iz2u()).transpose();
            Jres.bottomRows(1) = Jcon.transpose();
        }

        void _getStateIneqConstraints()
        {
            // TODO manage output bounds
            return;
        }

        void _getStateEqConstraints(
            bool hasGradient)
        {
            _ceq.setZero();
            _Jceq.setZero();

            static mat<AssignSize(sizeEnum::StateEqSize), AssignSize(sizeEnum::StateEqSize)> Jx;
            Jx.resize(GetSize(sizeEnum::StateEqSize), GetSize(sizeEnum::StateEqSize));
            Jx.setZero();

            // TODO support measured noise
            static mat<AssignSize(sizeEnum::StateEqSize), AssignSize(sizeEnum::InputPredictionSize)> Jmv;
            Jmv.resize(GetSize(sizeEnum::StateEqSize), GetSize(sizeEnum::InputPredictionSize));
            Jmv.setZero();

            static cvec<AssignSize(sizeEnum::StateEqSize)> Je;
            Je.resize(GetSize(sizeEnum::StateEqSize));
            Je.setZero();

            int ic = 0;

            static mat<Tnx, Tnx> Ix;
            Ix.resize(_dimensions.tnx, _dimensions.tnx);
            Ix.setIdentity(_dimensions.tnx, _dimensions.tnx);

            // TODO support scaling
            static mat<Tnx, Tnx> Sx, Tx;
            Sx.resize(_dimensions.tnx, _dimensions.tnx);
            Tx.resize(_dimensions.tnx, _dimensions.tnx);
            Sx.setIdentity(_dimensions.tnx, _dimensions.tnx);
            Tx.setIdentity(_dimensions.tnx, _dimensions.tnx);

            // TODO bind for continuos time
            if (_ctime)
            {
                //#pragma omp parallel for
                for (int i = 0; i < _dimensions.tph; i++)
                {
                    static cvec<Tnu> uk;
                    uk = _Umat.row(i).transpose();
                    static cvec<Tnx> xk; 
                    xk = _Xmat.row(i).transpose();

                    double h = _ts / 2.0;
                    static cvec<Tnx> xk1; 
                    xk1 = _Xmat.row(i + 1).transpose();

                    static cvec<Tnx> fk;
                    fk.resize(_dimensions.tnx);

                    _fUser(fk, xk, uk);

                    static cvec<Tnx> fk1;
                    fk1.resize(_dimensions.tnx);

                    _fUser(fk1, xk1, uk);

                    _ceq.middleRows(ic, _dimensions.tnx) = xk + (h * (fk + fk1)) - xk1;
                    // TODO support scaling
                    _ceq.middleRows(ic, _dimensions.tnx) = _ceq.middleRows(ic, _dimensions.tnx) / 1.0;

                    if (hasGradient)
                    {
                        static mat<Tnx, Tnx> Ak;
                        Ak.resize(_dimensions.tnx, _dimensions.tnx);

                        static mat<Tnx, Tnu> Bk;
                        Bk.resize(_dimensions.tnx, _dimensions.tnu);

                        _computeStateEqJacobian(Ak, Bk, fk, xk, uk);

                        static mat<Tnx, Tnx> Ak1;
                        Ak1.resize(_dimensions.tnx, _dimensions.tnx);

                        static mat<Tnx, Tnu> Bk1;
                        Bk1.resize(_dimensions.tnx, _dimensions.tnu);

                        _computeStateEqJacobian(Ak1, Bk1, fk1, xk1, uk);

                        if (i > 0)
                        {
                            Jx.middleCols((i - 1) * _dimensions.tnx, _dimensions.tnx).middleRows(ic, _dimensions.tnx) = 
                                Ix + (h * Sx * Ak * Tx);
                        }

                        Jx.middleCols(i * _dimensions.tnx, _dimensions.tnx).middleRows(ic, _dimensions.tnx) = 
                            -Ix + (h * Sx * Ak1 * Tx);
                        Jmv.middleCols(i * _dimensions.tnu, _dimensions.tnu).middleRows(ic, _dimensions.tnx) = 
                            h * Sx * (Bk + Bk1);
                    }

                    ic += _dimensions.tnx;
                }
            }
            else
            {
                //#pragma omp parallel for
                for (int i = 0; i < _dimensions.tph; i++)
                {
                    static cvec<Tnu> uk;
                    uk = _Umat.row(i).transpose();
                    static cvec<Tnx> xk;
                    xk = _Xmat.row(i).transpose();

                    static cvec<Tnx> xk1;
                    xk1.resize(_dimensions.tnx);

                    _fUser(xk1, xk, uk);

                    _ceq.middleRows(ic, _dimensions.tnx) = _Xmat.row(i + 1).transpose() - xk1;
                    // TODO support scaling
                    _ceq.middleRows(ic, _dimensions.tnx) = _ceq.middleRows(ic, _dimensions.tnx) / 1.0;

                    if (hasGradient)
                    {
                        static mat<Tnx, Tnx> Ak;
                        Ak.resize(_dimensions.tnx, _dimensions.tnx);

                        static mat<Tnx, Tnu> Bk;
                        Bk.resize(_dimensions.tnx, _dimensions.tnu);

                        _computeStateEqJacobian(Ak, Bk, xk1, xk, uk);

                        Ak = Sx * Ak * Tx;
                        Bk = Sx * Bk;

                        Jx.middleCols(i * _dimensions.tnx, _dimensions.tnx).middleRows(ic, _dimensions.tnx) = Ix;
                        if (i > 0)
                        {
                            Jx.middleCols((i - 1) * _dimensions.tnx, _dimensions.tnx).middleRows(ic, _dimensions.tnx) = -Ak;
                        }
                        Jmv.middleCols(i * _dimensions.tnx, _dimensions.tnx).middleRows(ic, _dimensions.tnx) = -Bk;
                    }

                    ic += Tnx;
                }
            }

            if (hasGradient)
                _glueJacobian<AssignSize(sizeEnum::StateEqSize)>(_Jceq, Jx, Jmv, Je);
        }

        void _computeUserIneqJacobian(
            mat<Tineq, AssignSize(sizeEnum::StateEqSize)> &Jconx,
            mat<Tineq, AssignSize(sizeEnum::InputPredictionSize)> &Jconmv,
            cvec<Tineq> &Jcone,
            mat<AssignSize(sizeEnum::TphPlusOne), Tnx> x0,
            mat<AssignSize(sizeEnum::TphPlusOne), Tnu> u0,
            double e0, cvec<Tineq> f0)
        {
            double dv = 1e-6;

            Jconx.setZero();

            // TODO support measured disturbaces
            Jconmv.setZero();

            Jcone.setZero();

            static mat<AssignSize(sizeEnum::TphPlusOne), Tnx> Xa;
            Xa = x0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Xa.rows(); i++)
            {
                for (int j = 0; j < (int)Xa.cols(); j++)
                {
                    Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
                }
            }

            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tph; i++)
            {
                for (int j = 0; j < _dimensions.tnx; j++)
                {
                    int ix = i + 1;
                    double dx = dv * Xa(j, 0);
                    x0(ix, j) = x0(ix, j) + dx;
                    static cvec<Tineq> f;
                    f.resize(_dimensions.tineq);
                    _ieqUser(f, x0, u0, _e);
                    x0(ix, j) = x0(ix, j) - dx;
                    static cvec<Tineq> df;
                    df = (f - f0) / dx;
                    Jconx.middleCols(i * _dimensions.tnx, _dimensions.tnx).col(j) = df;
                }
            }

            static mat<AssignSize(sizeEnum::TphPlusOne), Tnu> Ua;
            Ua = u0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Ua.rows(); i++)
            {
                for (int j = 0; j < (int)Ua.cols(); j++)
                {
                    Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
                }
            }

            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tph - 1; i++)
                // TODO support measured disturbaces
                for (int j = 0; j < _dimensions.tnu; j++)
                {
                    int k = j;
                    double du = dv * Ua(k, 0);
                    u0(i, k) = u0(i, k) + du;
                    static cvec<Tineq> f;
                    f.resize(_dimensions.tineq);
                    _ieqUser(f, x0, u0, _e);
                    u0(i, k) = u0(i, k) - du;
                    static cvec<Tineq> df;
                    df = (f - f0) / du;
                    Jconmv.middleCols(i * _dimensions.tnu, _dimensions.tnu).col(j) = df;
                }

            // TODO support measured disturbaces
            //#pragma omp parallel for
            for (int j = 0; j < _dimensions.tnu; j++)
            {
                int k = j;
                double du = dv * Ua(k, 0);
                u0(_dimensions.tph - 1, k) = u0(_dimensions.tph - 1, k) + du;
                u0(_dimensions.tph, k) = u0(_dimensions.tph, k) + du;
                static cvec<Tineq> f;
                f.resize(_dimensions.tineq);
                _ieqUser(f, x0, u0, _e);
                u0(_dimensions.tph - 1, k) = u0(_dimensions.tph - 1, k) - du;
                u0(_dimensions.tph, k) = u0(_dimensions.tph, k) - du;
                static cvec<Tineq> df;
                df = (f - f0) / du;
                Jconmv.middleCols((_dimensions.tph - 1) * _dimensions.tnu, _dimensions.tnu).col(j) = df;
            }

            double ea = fmax(1e-6, abs(e0));
            double de = ea * dv;
            static cvec<Tineq> f1;
            f1.resize(_dimensions.tineq);
            _ieqUser(f1, x0, u0, e0 + de);
            static cvec<Tineq> f2;
            f2.resize(_dimensions.tineq);
            _ieqUser(f2, x0, u0, e0 - de);
            Jcone = (f1 - f2) / (2 * de);
        }

        void _computeUserEqJacobian(
            mat<Teq, AssignSize(sizeEnum::StateEqSize)> &Jconx,
            mat<Teq, AssignSize(sizeEnum::InputPredictionSize)> &Jconmv,
            mat<AssignSize(sizeEnum::TphPlusOne), Tnx> x0,
            mat<AssignSize(sizeEnum::TphPlusOne), Tnu> u0, cvec<Teq> f0)
        {
            double dv = 1e-6;

            Jconx.setZero();

            // TODO support measured disturbaces
            Jconmv.setZero();

            static mat<AssignSize(sizeEnum::TphPlusOne), Tnx> Xa;
            Xa = x0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Xa.rows(); i++)
            {
                for (int j = 0; j < (int)Xa.cols(); j++)
                {
                    Xa(i, j) = (Xa(i, j) < 1) ? 1 : Xa(i, j);
                }
            }

            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tph; i++)
            {
                for (int j = 0; j < _dimensions.tnx; j++)
                {
                    int ix = i + 1;
                    double dx = dv * Xa(j, 0);
                    x0(ix, j) = x0(ix, j) + dx;
                    static cvec<Teq> f;
                    f.resize(_dimensions.teq);
                    _eqUser(f, x0, u0);
                    x0(ix, j) = x0(ix, j) - dx;
                    static cvec<Teq> df;
                    df = (f - f0) / dx;
                    Jconx.middleCols(i * _dimensions.tnx, _dimensions.tnx).col(j) = df;
                }
            }

            static mat<AssignSize(sizeEnum::TphPlusOne), Tnu> Ua;
            Ua = u0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < (int)Ua.rows(); i++)
            {
                for (int j = 0; j < (int)Ua.cols(); j++)
                {
                    Ua(i, j) = (Ua(i, j) < 1) ? 1 : Ua(i, j);
                }
            }

            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tph - 1; i++)
            {
                // TODO support measured disturbaces
                for (int j = 0; j < _dimensions.tnu; j++)
                {
                    int k = j;
                    double du = dv * Ua(k, 0);
                    u0(i, k) = u0(i, k) + du;
                    static cvec<Teq> f;
                    f.resize(_dimensions.teq);
                    _eqUser(f, x0, u0);
                    u0(i, k) = u0(i, k) - du;
                    static cvec<Teq> df;
                    df = (f - f0) / du;
                    Jconmv.middleCols(i * _dimensions.tnu, _dimensions.tnu).col(j) = df;
                }
            }

            // TODO support measured disturbaces
            //#pragma omp parallel for
            for (int j = 0; j < _dimensions.tnu; j++)
            {
                int k = j;
                double du = dv * Ua(k, 0);
                u0(_dimensions.tph - 1, k) = u0(_dimensions.tph - 1, k) + du;
                u0(_dimensions.tph, k) = u0(_dimensions.tph, k) + du;
                static cvec<Teq> f;
                f.resize(_dimensions.teq);
                _eqUser(f, x0, u0);
                u0(_dimensions.tph - 1, k) = u0(_dimensions.tph - 1, k) - du;
                u0(_dimensions.tph, k) = u0(_dimensions.tph, k) - du;
                static cvec<Teq> df;
                df = (f - f0) / du;
                Jconmv.middleCols((_dimensions.tph - 1) * _dimensions.tnu, _dimensions.tnu).col(j) = df;
            }
        }

        void _computeStateEqJacobian(
            mat<Tnx, Tnx> &Jx,
            mat<Tnx, Tnu> &Jmv,
            cvec<Tnx> f0,
            cvec<Tnx> x0,
            cvec<Tnu> u0)
        {
            Jx.setZero();
            Jmv.setZero();

            double dv = 1e-6;

            static cvec<Tnx> Xa;
            Xa = x0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tnx; i++)
            {
                Xa(i) = (Xa(i) < 1) ? 1 : Xa(i);
            }

            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tnx; i++)
            {
                double dx = dv * Xa(i);
                x0(i) = x0(i) + dx;
                static cvec<Tnx> f;
                f.resize(_dimensions.tnx);
                _fUser(f, x0, u0);
                x0(i) = x0(i) - dx;
                static cvec<Tnx> df;
                df = (f - f0) / dx;
                Jx.block(0, i, _dimensions.tnx, 1) = df;
            }

            cvec<Tnu> Ua = u0.cwiseAbs();
            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tnu; i++)
            {
                Ua(i) = (Ua(i) < 1) ? 1 : Ua(i);
            }

            //#pragma omp parallel for
            for (int i = 0; i < _dimensions.tnu; i++)
            {
                // TODO support measured disturbaces
                int k = i;
                double du = dv * Ua(k);
                u0(k) = u0(k) + du;
                static cvec<Tnx> f;
                f.resize(_dimensions.tnx);
                _fUser(f, x0, u0);
                u0(k) = u0(k) - du;
                static cvec<Tnx> df;
                df = (f - f0) / du;
                Jmv.block(0, i, _dimensions.tnx, 1) = df;
            }
        }

        int _nx;

        cvec<AssignSize(sizeEnum::StateEqSize)> _ceq;
        mat<AssignSize(sizeEnum::DecVarsSize), AssignSize(sizeEnum::StateEqSize)> _Jceq;

        cvec<AssignSize(sizeEnum::StateIneqSize)> _cineq;
        mat<AssignSize(sizeEnum::DecVarsSize), AssignSize(sizeEnum::StateIneqSize)> _Jcineq;

        cvec<Teq> _ceq_user;
        mat<AssignSize(sizeEnum::DecVarsSize), Teq> _Jceq_user;

        cvec<Tineq> _cineq_user;
        mat<AssignSize(sizeEnum::DecVarsSize), Tineq> _Jcineq_user;

        typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::StateFunHandle _fUser = nullptr;
        typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::IConFunHandle _ieqUser = nullptr;
        typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::EConFunHandle _eqUser = nullptr;
        typename BaseFunction<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq>::OutFunHandle _outUser = nullptr;
    };
} // namespace mpc
