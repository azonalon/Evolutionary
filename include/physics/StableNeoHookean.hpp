#pragma once
#include <Eigen/Dense>
#include <cassert>
#include "util/Math.hpp"

class StableNeoHookeanModel {
 public:
  static Eigen::Matrix<double, 6, 1> computeStressDifferentialComponents(
      double sx, double sy, double l, double mu, double eps);
  static Eigen::Matrix<double, 3, 1> computeStressGradientComponents(
      double sx, double sy, double l, double mu, double eps);

  static Eigen::Matrix2d JpF(const Eigen::Matrix2d& F) {
    Eigen::Matrix2d r;
    r << F(1, 1), -F(1, 0), -F(0, 1), F(0, 0);
    return r;
  }
  static inline double dDet(const Eigen::Matrix2d& F,
                            const Eigen::Matrix2d& dF) {
    return dF(1, 1) * F(0, 0) - dF(1, 0) * F(0, 1) - dF(0, 1) * F(1, 0) +
           dF(0, 0) * F(1, 1);
  }

  static double computeStressTensor(const Eigen::Matrix2d& F, double lambda,
                                    double mu, Eigen::Matrix2d& dest) {
    double alpha = 1 + mu / lambda - (mu / 4) * lambda;
    double Ic = (F.transpose() * F).trace();
    double detF = F.determinant();
    dest = mu * F * (1 - 1 / (Ic + 1)) + lambda * (detF - alpha) * JpF(F);
    return mu / 2 * (Ic - 2) + lambda / 2 * Math::sqrd(detF - alpha) -
           mu / 2 * log(Ic + 1);
  }

  static void computeStressDifferential(const Eigen::Matrix2d& F,
                                        const Eigen::Matrix2d& dF,
                                        double lambda, double mu,
                                        Eigen::Matrix2d& dest) {
    double alpha = 1 + mu / lambda - (mu / 4) * lambda;
    double Ic = (F.transpose() * F).trace();
    double detF = F.determinant();

    dest = 1 * mu * dF * (1 - 1 / (Ic + 1)) +
           1 * mu * F * (2 * (F.transpose() * dF).trace() / Math::sqrd(Ic + 1)) +
           1 * lambda * dDet(F, dF) * JpF(F) + 
           1 * lambda * (detF - alpha) * JpF(dF);
  }
};
