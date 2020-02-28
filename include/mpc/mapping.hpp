#pragma once

#include <mpc/mpc.hpp>

namespace mpc {
template <std::size_t Tnx, std::size_t Tnu, std::size_t Tph, std::size_t Tch>
class Common {
public:
    Common()
    {
        computeMapping();
    }

    void unwrapVector(const cvec<DecVarsSize> x, const cvec<Tnx> x0, mat<Tph + 1, Tnx>& Xmat, mat<Tph + 1, Tnu>& Umat, double& slack)
    {
        cvec<Tnu* Tch> u_vec = x.middleRows(Tph * Tnx, Tnu * Tch);

        mat<Tph + 1, Tnu> Umv;
        Umv.setZero();
        Umv.middleRows(0, Tph) = Iz2u * u_vec;
        Umv.row(Tph) = Umv.row(Tph - 1);

        Xmat.setZero();
        Xmat.row(0) = x0.transpose();
        for (size_t i = 1; i < Tph + 1; i++) {
            Xmat.row(i) = x.middleRows(((i - 1) * Tnx), Tnx).transpose();
            // TODO add rows scaling
        }

        // TODO add disturbaces manipulated vars
        Umat.setZero();
        Umat.block(0, 0, Tph + 1, Tnu) = Umv;

        slack = x[x.size() - 1];
    }

    mat<Tph * Tnu, Tch * Tnu> Iz2u;
    mat<Tch * Tnu, Tph * Tnu> Iu2z;
    mat<Tnu, Tnu> Sz2u;
    mat<Tnu, Tnu> Su2z;

private:
    void computeMapping()
    {
        cvec<Tch> m;
        for (size_t i = 0; i < Tch; i++) {
            m[i] = 1;
        }
        m[Tch - 1] = Tph - Tch + 1;

        Iz2u.setZero();
        Iu2z = Iz2u.transpose();

        Sz2u.setZero();
        Su2z.setZero();
        for (int i = 0; i < Sz2u.rows(); ++i) {
            // TODO add scaling factor
            double scale = 1.0;
            Sz2u(i, i) = scale;
            Su2z(i, i) = 1.0 / scale;
        }

        // TODO implement linear interpolation
        int ix = 0;
        int jx = 0;
        for (size_t i = 0; i < Tch; i++) {
            Iu2z.block(ix, jx, Tnu, Tnu) = Su2z;
            for (size_t j = 0; j < m[i]; j++) {
                Iz2u.block(jx, ix, Tnu, Tnu) = Sz2u;
                jx += Tnu;
            }
            ix += Tnu;
        }
    }
};
} // namespace mpc