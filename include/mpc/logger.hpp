#pragma once

#include <iostream>
#include <sstream>
#include <string>

//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#ifdef ERROR
#undef ERROR
#endif

namespace mpc {

class Logger {
public:
    enum level { DEEP,
        INFO, NONE, ERROR };

    static std::string prefix;
    static level logLevel;

    Logger() = delete;
    Logger(level type)
    {
        cl = type;
    }

    template <typename T>
    Logger& operator<<(const T& x)
    {
        if (logLevel <= level::NONE) {
            s << x;
        }
        return *this;
    }

    Logger& operator<<(std::ostream& (*f)(std::ostream& o))
    {
        s << f;
        return *this;
    };

    ~Logger()
    {
        if (logLevel <= cl) {
            std::cout << "[NLMPC";
            if(!prefix.empty()) {
                std::cout << " " << prefix << "] ";
            }
            else{
                std::cout << "] ";
            }
            std::cout << s.str();
        }
    }

private:
    std::stringstream s;
    level cl;
};

std::string Logger::prefix = "";
Logger::level Logger::logLevel = Logger::level::INFO;

} // namespace mpc