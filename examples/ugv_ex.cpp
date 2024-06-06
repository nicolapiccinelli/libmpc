#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>

struct Obstacle
{
    mpc::cvec<2> pos;
    double radius;
};

int main()
{
    constexpr int n_obs = 2;

    constexpr int Tnx = 4;
    constexpr int Tnu = 2;
    constexpr int Tny = 4;
    constexpr int Tph = 10;
    constexpr int Tch = 10;
    constexpr int Tineq = (Tph + 1) * n_obs;
    constexpr int Teq = 0;

    // list of n_obs obstacles
    Obstacle obs[n_obs];
    mpc::cvec<2> yref, v_pref;
    mpc::cvec<Tnx> m_x, m_x_next;

    double Ts = 0.1;

    mpc::NLMPC<Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq> controller;
    controller.setLoggerLevel(mpc::Logger::log_level::NORMAL);

    mpc::mat<Tnx, Tnx> A(Tnx, Tnx);
    mpc::mat<Tnx, Tnu> B(Tnx, Tnu);
    mpc::mat<Tny, Tnx> C(Tny, Tnx);
    mpc::mat<Tny, Tnu> D(Tny, Tnu);

    double m = 1.0;
    double r = 0.15;
    double speed = 1.0;

    A.setZero();
    A.block(0, 2, 2, 2).setIdentity();

    B.setZero();
    B.block(2, 0, 2, 2).setIdentity();
    B = B * 1.0/m;

    C.setIdentity();
    D.setZero();

    // discrete time model of the system
    mpc::mat<Tnx, Tnx> Ad(Tnx, Tnx);
    mpc::mat<Tnx, Tnu> Bd(Tnx, Tnu);
    mpc::mat<Tny, Tnx> Cd(Tny, Tnx);
    mpc::mat<Tny, Tnu> Dd(Tny, Tnu);

    mpc::discretization<Tnx,Tnu,Tny>(A, B, C, D, Ts, Ad, Bd, Cd, Dd);

    auto stateEq = [&](
                       mpc::cvec<Tnx> &dx,
                       const mpc::cvec<Tnx> &x,
                       const mpc::cvec<Tnu> &u,
                       const unsigned int &)
    {
        dx = Ad * x + Bd * u;
    };
    controller.setStateSpaceFunction(stateEq);

    auto outEq = [&](
                     mpc::cvec<Tny> &y,
                     const mpc::cvec<Tnx> &x,
                     const mpc::cvec<Tnu> &u,
                     const unsigned int &)
    {
        y = Cd * x + Dd * u;
    };
    controller.setOutputFunction(outEq);

    auto objEq = [&](
                     const mpc::mat<Tph + 1, Tnx> &x,
                     const mpc::mat<Tph + 1, Tny> &y,
                     const mpc::mat<Tph + 1, Tnu> &u,
                     const double &e)
    {
        double cost = 0;
        for (int i = 0; i < Tph + 1; i++)
        {
            cost += 1e3 * (x.row(i).segment(2, 2).transpose() - v_pref).squaredNorm();
            cost += 1e-2 * u.row(i).squaredNorm();
        }

        cost += 1e-5 * e * e;
        
        return cost;
    };
    controller.setObjectiveFunction(objEq);

    auto conIneq = [&](
                       mpc::cvec<Tineq> &ineq,
                       const mpc::mat<Tph + 1, Tnx> &x,
                       const mpc::mat<Tph + 1, Tny> &y,
                       const mpc::mat<Tph + 1, Tnu> &u,
                       const double &)
    {
        int index = 0;
        for (int i = 0; i < Tph + 1; i++)
        {
            for (size_t j = 0; j < n_obs; j++)
            {
                mpc::cvec<2> r_pos = x.row(i).segment(0,2).transpose() - obs[j].pos;
                ineq(index++) = obs[j].radius - r_pos.norm();
            }
        }
    };
    controller.setIneqConFunction(conIneq);

    // set current state
    m_x.setZero();
    m_x_next.setZero();

    obs[0].pos << 2.0, 1.0;
    obs[0].radius = 0.3;

    obs[1].pos << 1.0, 1.0;
    obs[1].radius = 0.3;

    // set the reference position
    yref << 2.0, 2.0;

    mpc::NLParameters params;
    params.maximum_iteration = 100;
    params.relative_ftol = -1;
    params.relative_xtol = -1;
    params.hard_constraints = false;
    params.enable_warm_start = true;

    controller.setOptimizerParameters(params);

    auto res = controller.getLastResult();
    res.cmd.setZero();

    double t = 0;
    for(int i = 0; i < 150; i++)
    {
        // solve
        res = controller.optimize(m_x, res.cmd);
        
        // apply vector field
        stateEq(m_x_next, m_x, res.cmd, -1);
        m_x = m_x_next;
        t += Ts;

        std::cout << t << "," << m_x(0) << "," << m_x(1) << "," << m_x(2) << "," << m_x(3);
        std::cout << "," << yref(0) << "," << yref(1);
        std::cout << "," << res.cmd(0) << "," << res.cmd(1);
        std::cout << "," << res.cost << "," << res.solver_status;
        std::cout << std::endl;

        v_pref = (yref - m_x.segment(0, 2)).normalized() * speed;

        // break the loop if the distance between m_x and yref is than a threshold
        if((m_x.segment(0,2) - yref).norm() < 0.05)
        {
            break;
        }
    }

    std::cout << controller.getExecutionStats();

    return 0;
}