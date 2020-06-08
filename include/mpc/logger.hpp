#pragma once

#include <iostream>
#include <ostream>
#include <string>

//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */

#ifdef ERROR
#undef ERROR
#endif

namespace mpc
{
    class Logger
    {
    public:
        enum log_type
        {
            DEBUG = 0,
            INFO = 1,
            ERROR = 2
        };

        enum log_level
        {
            DEEP = 0,
            NORMAL = 1,
            ALERT = 2,
            NONE
        };

        static Logger &instance()
        {
            static Logger instance;
            return instance;
        }

        Logger &reset()
        {
            resetImpl();
            return *this;
        }

        Logger &log(log_type type)
        {
            Logger::instance().currentType = type;
            if (Logger::instance().thresholdLevel <= Logger::instance().currentType)
            {
                *(Logger::instance().os) << "[NLMPC";
                if (!Logger::instance().prefix.empty())
                {
                    *(Logger::instance().os) << " " << Logger::instance().prefix << "] ";
                }
                else
                {
                    *(Logger::instance().os) << "] ";
                }
            }

            return *this;
        }

        Logger &setStream(std::ostream *opt_stream)
        {
            os = opt_stream;
            return *this;
        }

        Logger &setLevel(log_level l)
        {
            thresholdLevel = l;
            return *this;
        }

        Logger &setPrefix(std::string s)
        {
            prefix = s;
            return *this;
        }

        template <typename T>
        Logger &operator<<(const T &x)
        {
            if (thresholdLevel <= currentType)
            {
                *os << x;
            }

            return *this;
        }

        Logger &operator<<(std::ostream &(*f)(std::ostream &o))
        {
            if (thresholdLevel <= currentType)
            {
                *os << f;
            }

            return *this;
        };

    private:
        Logger() : os(&std::cout)
        {
            resetImpl();
        }

        void resetImpl()
        {
            prefix = "";
            thresholdLevel = log_level::NORMAL;
        }

        Logger(const Logger &) = delete;
        Logger &operator=(const Logger &) = delete;

        std::ostream *os;
        std::string prefix;
        log_level thresholdLevel;
        log_type currentType;
    };

} // namespace mpc