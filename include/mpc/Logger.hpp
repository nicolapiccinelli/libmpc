/*
 *   Copyright (c) 2023-2025 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <iostream>
#include <ostream>
#include <string>
#include <cstdlib> // for std::getenv

#ifdef ERROR
#undef ERROR
#endif

namespace mpc
{
    /**
     * @brief Basic logging system for different log levels and types.
     */
    class Logger
    {
    public:
        /**
         * @brief Enumerates the different log types.
         */
        enum LogType
        {
            DETAIL = 0,
            INFO = 1,
            ERROR = 2
        };

        /**
         * @brief Enumerates the external log levels that control the verbosity.
         */
        enum LogLevel
        {
            UNSET = -1, ///< Unset log level
            DEEP = 0,   ///< Deep logging with maximum verbosity
            NORMAL = 1, ///< Normal logging
            ALERT = 2,  ///< Only alert-level messages
            NONE = 3    ///< No logging
        };

        /**
         * @brief Gets the singleton instance of the Logger.
         * @return Logger& The logger instance.
         */
        static Logger &instance()
        {
            static Logger instance;
            return instance;
        }

        /**
         * @brief Resets the logger configuration to its default state.
         * @return Logger& The updated logger instance.
         */
        Logger &reset()
        {
            prefix.clear();
            thresholdLevel = LogLevel::NORMAL;
            verboseOverride = false;

            if (const char *env_p = std::getenv("MPCXX_LOG_LEVEL_OVERRIDE"))
            {
                verboseOverride = true;

                thresholdLevelOverride = parseLogLevel(env_p);
                
                if (thresholdLevelOverride == LogLevel::UNSET)
                {
                    verboseOverride = false;
                }
            }

            return *this;
        }

        /**
         * @brief Sets the log type for the next log message.
         * @param type The log type (e.g., INFO, ERROR).
         * @return Logger& The updated logger instance.
         */
        Logger &log(LogType type)
        {
            currentType = type;

            if (shouldLog())
            {
                writePrefix();
            }
            return *this;
        }

        /**
         * @brief Sets the output stream for the logger (e.g., std::cout or file stream).
         * @param outputStream The output stream.
         * @return Logger& The updated logger instance.
         */
        Logger &setStream(std::ostream *outputStream)
        {
            os = outputStream;
            return *this;
        }

        /**
         * @brief Sets the log level that controls the verbosity of log messages.
         * @param level The log level.
         * @return Logger& The updated logger instance.
         */
        Logger &setLevel(LogLevel level)
        {
            thresholdLevel = level;
            return *this;
        }

        /**
         * @brief Sets a custom prefix for log messages.
         * @param prefix The custom prefix string.
         * @return Logger& The updated logger instance.
         */
        Logger &setPrefix(const std::string &prefix)
        {
            this->prefix = prefix;
            return *this;
        }

        /**
         * @brief Writes the log message to the output stream.
         * @param message The message to log.
         * @return Logger& The updated logger instance.
         */
        template <typename T>
        Logger &operator<<(const T &message)
        {
            if (shouldLog())
            {
                *os << message;
            }
            return *this;
        }

        /**
         * @brief Handles formatting of the log stream.
         * @param manip The manipulator function (e.g., std::endl).
         * @return Logger& The updated logger instance.
         */
        Logger &operator<<(std::ostream &(*manip)(std::ostream &))
        {
            if (shouldLog())
            {
                *os << manip;
            }
            return *this;
        }

    private:
        Logger() : os(&std::cout)
        {
            reset();
        }

        /**
         * @brief Determines whether the current log message should be printed based on the threshold level.
         * @return true if the message should be logged, false otherwise.
         */
        bool shouldLog() const
        {
            int activeLevel = verboseOverride ? (int)thresholdLevelOverride : (int)thresholdLevel;
            return activeLevel <= (int)currentType;
        }

        /**
         * @brief Parses a log level string from the environment variable.
         * @param level The environment variable value.
         * @return LogLevel The corresponding log level.
         */
        LogLevel parseLogLevel(const std::string &level)
        {
            if (level == "DEEP")
            {
                return LogLevel::DEEP;
            }

            if (level == "NORMAL")
            {
                return LogLevel::NORMAL;
            }

            if (level == "ALERT")
            {
                return LogLevel::ALERT;
            }

            if (level == "NONE")
            {
                return LogLevel::NONE;
            }
            
            return LogLevel::UNSET;
        }

        /**
         * @brief Writes the log prefix (if set) to the output stream.
         */
        void writePrefix()
        {
            *os << "[MPC++";
            if (!prefix.empty())
            {
                *os << " " << prefix;
            }
            *os << "] ";
        }

        // Deleted copy constructor and assignment operator to prevent copying of the Logger.
        Logger(const Logger &) = delete;
        Logger &operator=(const Logger &) = delete;

        std::ostream *os;                                   ///< The output stream (default is std::cout).
        std::string prefix;                                 ///< The custom prefix for log messages.
        bool verboseOverride = false;                       ///< Flag indicating whether environment variable overrides are active.
        LogLevel thresholdLevel = LogLevel::NORMAL;         ///< The default log level threshold.
        LogLevel thresholdLevelOverride = LogLevel::NORMAL; ///< The log level override from the environment.
        LogType currentType = LogType::INFO;                ///< The current log type.
    };

} // namespace mpc
