#pragma once
#include <mpc/Types.hpp>

namespace mpc {

/**
 * @brief Problem dimension accessor system 
 * @tparam T 
 */
template <int T>
class Dim {
public:
    Dim() = default;

    /**
     * @brief Sum two dimensions, in case of static or dynamic allocation 
     * it automatically propagates the right object
     * 
     * @tparam W static dimension of the other operand
     * @return constexpr auto instance of the result dimension
     */
    template <int W>
    constexpr auto operator+(const Dim<W>&)
    {
        return Dim<makeDim(T + W, isStatic<T,W>())>();
    }

    /**
     * @brief Subtract two dimensions, in case of static or dynamic allocation 
     * it automatically propagates the right object
     * 
     * @tparam W static dimension of the other operand
     * @return constexpr auto instance of the result dimension
     */
    template <int W>
    constexpr auto operator-(const Dim<W>&)
    {
        return Dim<makeDim(T - W, isStatic<T, W>())>();
    }

    /**
     * @brief Multiply two dimensions, in case of static or dynamic allocation 
     * it automatically propagates the right object
     * 
     * @tparam W static dimension of the other operand
     * @return constexpr auto instance of the result dimension
     */
    template <int W>
    constexpr auto operator*(const Dim<W>&)
    {
        return Dim<makeDim(T * W, isStatic<T, W>())>();
    }

    /**
     * @brief Override the int typecasting to provide the template argument 
     * automatically during assigments
     * 
     * @return int the template class argument
     */
    constexpr operator int() { return T; }

    /**
     * @brief Return the value of the static dimension, in case of dynamic allocation
     * this value would be -1. This function should be used only to define container dimension,
     * to get the right dimension use the num() function instead
     * 
     * @return constexpr int the template class argument
     */
    constexpr int get()
    {
        return T;
    }

    /**
     * @brief Return the value of the runtime dimension
     * 
     * @return size_t 
     */
    size_t num()
    {
        return innerDimension;
    }

    /**
     * @brief Set the runtime dimension
     * 
     * @param d dimension to be assigned at runtime
     */
    void setDynDim(size_t d)
    {
        innerDimension = d;
    }

private:
    /**
     * @brief Check whenever the operator should create a static
     * or dynamic dimension object instance
     * 
     * @tparam A first dimension
     * @tparam B second dimension
     * @return true 
     * @return false 
     */
    template <int A, int B>
    constexpr static bool isStatic()
    {
        return A > Eigen::Dynamic && B > Eigen::Dynamic;
    }

    size_t innerDimension;
};

} // namespace mpc
