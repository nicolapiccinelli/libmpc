#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace mpc {
class Logger {
public:
    enum level { DEEP,
        INFO };

    static int verbose;
    static level logLevel;

    Logger() = delete;
    Logger(level type)
    {
        cl = type;
    }

    template <typename T>
    Logger& operator<<(const T& x)
    {
        if (verbose) {
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
        if (verbose && logLevel <= cl) {
            std::cout << "[NLMPC] " << s.str();
        }
    }

private:
    std::stringstream s;
    level cl;
};
} // namespace mpc