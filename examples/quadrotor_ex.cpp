#include <mpc/LMPC.hpp>

int main()
{
    constexpr int Tnx = 12;
    constexpr int Tny = 12;
    constexpr int Tnu = 4;
    constexpr int Tndu = 4;
    constexpr int Tph = 10;
    constexpr int Tch = 10;

    mpc::LMPC<
    Tnx, Tnu, Tndu, Tny,
    Tph, Tch>
    controller;

    controller.setLoggerLevel(mpc::Logger::log_level::NORMAL);

    mpc::mat<Tnx, Tnx> Ad;
    Ad << 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0,
    0.0488, 0, 0, 1, 0, 0, 0.0016, 0, 0, 0.0992, 0, 0,
    0, -0.0488, 0, 0, 1, 0, 0, -0.0016, 0, 0, 0.0992, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0992,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0.9734, 0, 0, 0, 0, 0, 0.0488, 0, 0, 0.9846, 0, 0,
    0, -0.9734, 0, 0, 0, 0, 0, -0.0488, 0, 0, 0.9846, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9846;

    mpc::mat<Tnx, Tnu> Bd;
    Bd << 0, -0.0726, 0, 0.0726,
    -0.0726, 0, 0.0726, 0,
    -0.0152, 0.0152, -0.0152, 0.0152,
    0, -0.0006, -0.0000, 0.0006,
    0.0006, 0, -0.0006, 0,
    0.0106, 0.0106, 0.0106, 0.0106,
    0, -1.4512, 0, 1.4512,
    -1.4512, 0, 1.4512, 0,
    -0.3049, 0.3049, -0.3049, 0.3049,
    0, -0.0236, 0, 0.0236,
    0.0236, 0, -0.0236, 0,
    0.2107, 0.2107, 0.2107, 0.2107;

    mpc::mat<Tny, Tnx> Cd;
    Cd.setIdentity();

    controller.setStateSpaceModel(Ad, Bd, Cd);
    
    mpc::cvec<Tnu> InputW, DeltaInputW;
    mpc::cvec<Tny> OutputW;

    OutputW << 0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5;
    InputW << 0.1, 0.1, 0.1, 0.1;
    DeltaInputW << 0, 0, 0, 0;

    controller.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, Tph});

    mpc::cvec<Tnx> xmin, xmax;
    xmin << -M_PI / 6, -M_PI / 6, -mpc::inf, -mpc::inf, -mpc::inf, -1,
    -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf;

    xmax << M_PI / 6, M_PI / 6, mpc::inf, mpc::inf, mpc::inf, mpc::inf,
    mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf;

    mpc::cvec<Tny> ymin, ymax;
    ymin.setOnes();
    ymin *= -mpc::inf;
    ymax.setOnes();
    ymax *= mpc::inf;

    mpc::cvec<Tnu> umin, umax;
    double u0 = 10.5916;
    umin << 9.6, 9.6, 9.6, 9.6;
    umin.array() -= u0;
    umax << 13, 13, 13, 13;
    umax.array() -= u0;

    controller.setStateBounds(xmin, xmax, {0, Tph});
    controller.setOutputBounds(ymin, ymax, {0, Tph});
    controller.setInputBounds(umin, umax, {0, Tch});

    controller.setReferences(mpc::mat<Tny, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero());

    mpc::cvec<Tny> yRef;
    yRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    controller.setReferences(yRef, mpc::cvec<Tnu>::Zero(), mpc::cvec<Tnu>::Zero(), {0, Tph});

    mpc::LParameters params;
    params.maximum_iteration = 250;
    controller.setOptimizerParameters(params);

    auto res = controller.optimize(mpc::cvec<Tnx>::Zero(), mpc::cvec<Tnu>::Zero());
    auto seq = controller.getOptimalSequence();
    
    std::cout << "Optimal control input: " << res.cmd << std::endl;

    std::cout << "Optimal sequence (input): " << seq.input << std::endl;
    std::cout << "Optimal sequence (output): " << seq.output << std::endl;
    std::cout << "Optimal sequence (state): " << seq.state << std::endl;

    std::cout << controller.getExecutionStats();

    return 0;
}