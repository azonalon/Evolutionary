#pragma once
#include <Eigen/Dense>
#include <cassert>
#include "util/Math.hpp"

/**
 * Invertible neo-hookean stress tensor computation
 */

class InvertibleNeoHookeanModel {
public:
    Eigen::Matrix2d temp2x2A, temp2x2B, temp2x2P, temp2x2D, b;
    double eps = 0.3;

    InvertibleNeoHookeanModel(double eps) {
        assert(eps > 0.01 && eps < 1);
        this->eps = eps;
    }
    static Eigen::Matrix<double,6,1> computeStressDifferentialComponents(
        double sx, double sy, double l,
        double mu, double eps);
    static Eigen::Matrix<double,3,1> computeStressGradientComponents(
        double sx, double sy, double l,
        double mu, double eps);
    double computeStressTensor(const Eigen::Matrix2d& F,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& dest);

    void computeStressDifferential(const Eigen::Matrix2d& F, const Eigen::Matrix2d& dF,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& dest);
};
