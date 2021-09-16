#pragma once
#include <mpc/Types.hpp>

namespace mpc {

template <int T>
class Dim {
public:
    Dim() = default;

    template <int W>
    constexpr auto operator+(const Dim<W>&)
    {
        return Dim<makeDim(T + W, isStatic<T,W>())>();
    }

    template <int W>
    constexpr auto operator-(const Dim<W>&)
    {
        return Dim<makeDim(T - W, isStatic<T, W>())>();
    }

    template <int W>
    constexpr auto operator*(const Dim<W>&)
    {
        return Dim<makeDim(T * W, isStatic<T, W>())>();
    }

    constexpr operator int() { return T; }

    constexpr int get()
    {
        return T;
    }

    size_t num()
    {
        return innerDimension;
    }

    void setDynDim(size_t d)
    {
        innerDimension = d;
    }

private:
    template <int A, int B>
    constexpr static bool isStatic()
    {
        return A > Eigen::Dynamic && B > Eigen::Dynamic;
    }

    size_t innerDimension;
};

} // namespace mpc
