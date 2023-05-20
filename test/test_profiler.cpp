#include "basic.hpp"
#include <mpc/Profiler.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <thread>

TEST_CASE("SolutionStats", "[SolutionStats]")
{
    mpc::SolutionStats stats;

    SECTION("Clear function resets all statistical values")
    {
        stats.minSolutionTime = std::chrono::duration<double>(5.0);
        stats.maxSolutionTime = std::chrono::duration<double>(10.0);
        stats.averageSolutionTime = std::chrono::duration<double>(7.5);
        stats.totalSolutionTime = std::chrono::duration<double>(30.0);
        stats.standardDeviation = 1.5;
        stats.numberOfSolutions = 3;
        stats.solutionsStates = {
            {mpc::ResultStatus::SUCCESS, 2},
            {mpc::ResultStatus::ERROR, 1}};

        stats.clear();

        REQUIRE(stats.minSolutionTime == std::chrono::duration<double>::max());
        REQUIRE(stats.maxSolutionTime == std::chrono::duration<double>::min());
        REQUIRE(stats.averageSolutionTime == std::chrono::duration<double>::zero());
        REQUIRE(stats.totalSolutionTime == std::chrono::duration<double>::zero());
        REQUIRE(stats.standardDeviation == 0.0);
        REQUIRE(stats.numberOfSolutions == 0);
        REQUIRE(stats.solutionsStates.empty());
    }

    SECTION("resultStatusToString returns correct string representation")
    {
        REQUIRE(mpc::SolutionStats::resultStatusToString(mpc::ResultStatus::SUCCESS) == "SUCCESS");
        REQUIRE(mpc::SolutionStats::resultStatusToString(mpc::ResultStatus::MAX_ITERATION) == "MAX_ITERATION");
        REQUIRE(mpc::SolutionStats::resultStatusToString(mpc::ResultStatus::INFEASIBLE) == "INFEASIBLE");
        REQUIRE(mpc::SolutionStats::resultStatusToString(mpc::ResultStatus::ERROR) == "ERROR");
        REQUIRE(mpc::SolutionStats::resultStatusToString(mpc::ResultStatus::UNKNOWN) == "UNKNOWN");
        REQUIRE(mpc::SolutionStats::resultStatusToString(static_cast<mpc::ResultStatus>(99)) == "INVALID");
    }
}

TEST_CASE("Profiler", "[Profiler]")
{
    mpc::Profiler profiler;

    SECTION("Reset function clears profiler statistics")
    {
        profiler.solutionStart();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        profiler.solutionEnd(mpc::Result<1>());

        profiler.reset();

        const mpc::SolutionStats &stats = profiler.getStats();
        
        REQUIRE(stats.minSolutionTime == std::chrono::duration<double>::max());
        REQUIRE(stats.maxSolutionTime == std::chrono::duration<double>::min());
        REQUIRE(stats.averageSolutionTime == std::chrono::duration<double>::zero());
        REQUIRE(stats.totalSolutionTime == std::chrono::duration<double>::zero());
        REQUIRE(stats.standardDeviation == 0.0);
        REQUIRE(stats.numberOfSolutions == 0);
        REQUIRE(stats.solutionsStates.empty());
    }

    SECTION("solutionStart and solutionEnd update statistics correctly")
    {
        profiler.solutionStart();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        mpc::Result<1> res;
        res.status = mpc::ResultStatus::SUCCESS;
        std::chrono::duration<double> elapsedTime = profiler.solutionEnd(res);

        const mpc::SolutionStats &stats = profiler.getStats();

        REQUIRE(stats.minSolutionTime <= elapsedTime);
        REQUIRE(stats.maxSolutionTime >= elapsedTime);
        REQUIRE(stats.averageSolutionTime >= elapsedTime);
        REQUIRE(stats.totalSolutionTime >= elapsedTime);
        REQUIRE(stats.numberOfSolutions == 1);
        REQUIRE(stats.solutionsStates.at(mpc::ResultStatus::SUCCESS) == 1);
    }

    SECTION("Empty profiler has default statistics")
    {
        const mpc::SolutionStats &stats = profiler.getStats();

        REQUIRE(stats.minSolutionTime == std::chrono::duration<double>::max());
        REQUIRE(stats.maxSolutionTime == std::chrono::duration<double>::min());
        REQUIRE(stats.averageSolutionTime == std::chrono::duration<double>::zero());
        REQUIRE(stats.totalSolutionTime == std::chrono::duration<double>::zero());
        REQUIRE(stats.standardDeviation == 0.0);
        REQUIRE(stats.numberOfSolutions == 0);
        REQUIRE(stats.solutionsStates.empty());
    }

    SECTION("Profiler correctly calculates average solution time, standard deviation, number of solutions and solution states")
    {
        mpc::Result<1> res;
        profiler.solutionStart();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        res.status = mpc::ResultStatus::SUCCESS;
        profiler.solutionEnd(res);

        profiler.solutionStart();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        res.status = mpc::ResultStatus::MAX_ITERATION;
        profiler.solutionEnd(res);

        profiler.solutionStart();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        res.status = mpc::ResultStatus::SUCCESS;
        profiler.solutionEnd(res);

        const mpc::SolutionStats &stats = profiler.getStats();

        REQUIRE(stats.solutionsStates.at(mpc::ResultStatus::SUCCESS) == 2);
        REQUIRE(stats.solutionsStates.at(mpc::ResultStatus::MAX_ITERATION) == 1);
        REQUIRE(stats.averageSolutionTime.count() == Catch::Approx(0.2).margin(0.01));
        REQUIRE(stats.standardDeviation == Catch::Approx(0.081).margin(0.01));
        REQUIRE(stats.numberOfSolutions == 3);
    }
}
