/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <chrono>
#include <mpc/Types.hpp>

namespace mpc
{
    /**
     * \class SolutionStats
     * \brief Class for collecting solution statistics.
     *
     * The SolutionStats class is responsible for collecting and storing statistics related to solution execution.
     * It tracks the minimum solution time, maximum solution time, average solution time, total solution time,
     * standard deviation, number of solutions, and solution status percentages.
     */
    class SolutionStats
    {
    public:
        /**
         * \brief Constructor for the SolutionStats class.
         * 
         * This constructor initializes all statistical values to their default values.
         */
        SolutionStats()
        {
            clear();
        }

        /**
         * \brief Resets all statistical values to their default values.
         *
         * This function sets the minimum solution time to the maximum possible duration,
         * the maximum solution time to the minimum possible duration,
         * the average solution time to zero,
         * the total solution time to zero,
         * the standard deviation to zero,
         * the number of solutions to zero,
         * and clears the solutionsStates map.
         */
        void clear()
        {
            minSolutionTime = std::chrono::duration<double>::max();
            maxSolutionTime = std::chrono::duration<double>::min();
            averageSolutionTime = std::chrono::duration<double>::zero();
            totalSolutionTime = std::chrono::duration<double>::zero();
            standardDeviation = 0.0;
            numberOfSolutions = 0;

            solutionsStates.clear();
        }

        /**
         * \brief Converts a ResultStatus enum to its string representation.
         *
         * \param status The ResultStatus enum to convert.
         * \return The string representation of the ResultStatus enum.
         */
        static std::string resultStatusToString(ResultStatus status)
        {
            switch (status)
            {
            case ResultStatus::SUCCESS:
                return "SUCCESS";
            case ResultStatus::MAX_ITERATION:
                return "MAX_ITERATION";
            case ResultStatus::INFEASIBLE:
                return "INFEASIBLE";
            case ResultStatus::ERROR:
                return "ERROR";
            case ResultStatus::UNKNOWN:
                return "UNKNOWN";
            default:
                return "INVALID";
            }
        }

        /**
         * \brief Overloads the stream insertion operator to print SolutionStats to an output stream.
         *
         * \param os The output stream.
         * \param stats The SolutionStats object to print.
         * \return The output stream.
         */
        friend std::ostream &operator<<(std::ostream &os, const SolutionStats &stats)
        {
            os << std::fixed << std::setprecision(6);
            os << "Solution stats:" << std::endl;
            os << "  - min: " << stats.minSolutionTime.count() << " s" << std::endl;
            os << "  - max: " << stats.maxSolutionTime.count() << " s" << std::endl;
            os << "  - avg: " << stats.averageSolutionTime.count() << " s" << std::endl;
            os << "  - std: " << stats.standardDeviation << " s" << std::endl;

            if (!stats.solutionsStates.empty())
            {
                os << "Solution status percentages:" << std::endl;
                for (const auto &pair : stats.solutionsStates)
                {
                    double percentage = (static_cast<double>(pair.second) / stats.numberOfSolutions) * 100;
                    os << "  - " << SolutionStats::resultStatusToString(pair.first)
                       << ": " << std::fixed << std::setprecision(2)
                       << percentage << "%" << std::endl;
                }
            }

            os << "Total time and number of solutions:" << std::endl;
            os << "  - tot: " << stats.totalSolutionTime.count() << " s" << std::endl;
            os << "  - num: " << stats.numberOfSolutions << std::endl;

            return os;
        }

        std::chrono::duration<double> minSolutionTime, maxSolutionTime, averageSolutionTime, totalSolutionTime;
        std::map<ResultStatus, int> solutionsStates;
        double standardDeviation;
        int numberOfSolutions;
    };

    /**
     * \class Profiler
     * \brief Class for measuring solution execution time and collecting statistics.
     *
     * The Profiler class measures the execution time of a solution using a high-resolution clock.
     * It keeps track of the minimum, maximum, average, and standard deviation of the solution time,
     * as well as the number of solutions and the total time.
     */
    class Profiler
    {
    public:
        /**
         * \brief Default constructor.
         */
        Profiler() = default;

        /**
         * \brief Destructor.
         */
        ~Profiler() = default;

        /**
         * \brief Resets the profiler statistics.
         */
        void reset()
        {
            stats.clear();
        }

        /**
         * \brief Records the start time of a solution.
         */
        void solutionStart()
        {
            startTime = std::chrono::high_resolution_clock::now();
        }

        /**
         * \brief Ends a solution and returns the elapsed time.
         *
         * \tparam Tnu The size of the Result vector.
         * \param result The Result object containing the solution information.
         * \return The elapsed time for the solution as a std::chrono::duration<double>.
         */
        template <int Tnu>
        std::chrono::duration<double> solutionEnd(const Result<Tnu> &result)
        {
            auto stopTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration<double>(stopTime - startTime);

            // update the statistics about the result type
            stats.solutionsStates[result.status]++;

            addSolutionTime(elapsedTime);
            return elapsedTime;
        }

        /**
         * \brief Returns a reference to the solution statistics.
         *
         * \return A constant reference to the SolutionStats object.
         */
        const SolutionStats &getStats()
        {
            return stats;
        }

    private:
        /**
         * \brief Adds the elapsed time of a solution to the statistics.
         *
         * This function updates the statistics of the Profiler instance by adding the elapsed time of a solution.
         * It updates the minimum, maximum, total and average solution times,
         * as well as the number of solutions and the standard deviation.
         *
         * \param time The elapsed time of the solution.
         */
        void addSolutionTime(const std::chrono::duration<double> &time)
        {
            stats.totalSolutionTime += time;
            stats.numberOfSolutions++;

            if (time < stats.minSolutionTime)
            {
                stats.minSolutionTime = time;
            }

            if (time > stats.maxSolutionTime)
            {
                stats.maxSolutionTime = time;
            }

            if (stats.numberOfSolutions > 0)
            {
                stats.averageSolutionTime = (1.0 / stats.numberOfSolutions) * (time + (stats.numberOfSolutions - 1) * stats.averageSolutionTime);

                if(stats.numberOfSolutions == 1)
                {
                    stats.standardDeviation = 0.0;
                }
                else
                {
                    double residual = (stats.numberOfSolutions/(stats.numberOfSolutions - 1)) * (pow(time.count() - stats.averageSolutionTime.count(), 2));
                    double stdDev_old = (stats.numberOfSolutions - 2) * pow(stats.standardDeviation, 2);
                    stats.standardDeviation = sqrt((stdDev_old + residual) / (stats.numberOfSolutions - 1));
                }
            }
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
        SolutionStats stats;
    };
}