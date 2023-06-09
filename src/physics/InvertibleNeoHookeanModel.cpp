#include "physics/InvertibleNeoHookeanModel.hpp"
#include <iostream>


double InvertibleNeoHookeanModel::computeStressTensor(
    const Eigen::Matrix2d& F,
    double lambda, double mu,
    Eigen::Matrix2d& dest)
{
    // assert(F(0,1) == 0 && F(1,0) == 0);
    // assert(F(0,0) <= 1 && F(0,0) >= -1);
    double J = F.determinant();
    auto svd = Math::singularValueDecomposition(F);
    double u1 = svd[0], u2 = svd[1],
            sx = svd[2], sy = svd[3],
            v1 = svd[4], v2 = svd[5];
    if(J > eps && sx > 0) {
        // normal neohookean
        temp2x2P = F.transpose()*F;
        double I1 = temp2x2P.trace();
        double logJ = std::log(J);
        Math::invertTranspose(F, temp2x2P);
        dest = mu*F + (lambda*logJ - mu)*temp2x2P;
        return mu*(I1/2.0 - 1 - logJ) + lambda/2.0*logJ*logJ;
    } else {
        // extrapolate
        // std::cout << svd << std::endl;
#ifdef IFE_SOLVER
        assert(!(sx < 0 && sy < 0));
        if(sx < 0) {
            assert(-sx <= sy);
        }
        if(sy < 0) {
            assert(-sy <= sx);
        }
#endif
        auto r = computeStressGradientComponents(sx, sy, lambda, mu, eps);
        double psi0 = r[0], psi1=r[1], psi2=r[2];
        dest(0,0) = psi1*u1*v1 - psi2*u2*v2;
        dest(0,1) = -(psi2*u2*v1) - psi1*u1*v2;
        dest(1,0) = psi1*u2*v1 + psi2*u1*v2;
        dest(1,1) = psi2*u1*v1 - psi1*u2*v2;
        return psi0;
    }
}

void InvertibleNeoHookeanModel::computeStressDifferential(const Eigen::Matrix2d& F, const Eigen::Matrix2d& dF,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& dest)
{
    // double J = det(F);
    double J = F.determinant();
    if(J > eps) {
        double logJ = std::log(J);
        // normal neohookean
        Math::invertTranspose(F, temp2x2P);
        temp2x2D = temp2x2P*dF.transpose();
        double trIFdF = temp2x2D.trace();
        dest = temp2x2D*temp2x2P;
        dest = mu*dF + (mu-lambda*logJ)*dest;
        dest = dest + lambda*trIFdF*temp2x2P;
    } else {
        // extrapolate
        auto svd = Math::singularValueDecomposition(F);
        double u1 = svd[0], u2 = svd[1],
                sx = svd[2], sy = svd[3],
                v1 = svd[4], v2 = svd[5];
        auto r = computeStressDifferentialComponents(sx, sy, lambda, mu, eps);
        assert(std::abs(J-sx*sy) < 1e-9);
        // double psi0 = r[0];
        double f1=r[1], f2=r[2], psi11=r[3], psi12=r[4], psi22=r[5];
        temp2x2A(0,0)=u1;temp2x2A(0,1)=-u2;
        temp2x2A(1,0)=u2;temp2x2A(1,1)=u1;
        temp2x2B(0,0)=v1;temp2x2B(0,1)=-v2;
        temp2x2B(1,0)=v2;temp2x2B(1,1)=v1;
        // b = dF*temp2x2B.transpose();
        temp2x2D = temp2x2A.transpose()*dF;
        b = temp2x2D*temp2x2B.transpose();
        temp2x2D(0,0) = b(0,0)*psi11 + b(1,1)*psi12;
        temp2x2D(0,1) = b(1,0)*f1 + b(0,1)*f2;
        temp2x2D(1,0) = b(0,1)*f1 + b(1,0)*f2;
        temp2x2D(1,1) = b(0,0)*psi12 + b(1,1)*psi22;
        temp2x2P = temp2x2A*temp2x2D;
        dest = temp2x2P*temp2x2B;
    }
}
