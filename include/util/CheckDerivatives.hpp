#include "Math.hpp"
#include "stdio.h"
#include "physics/ImplicitODESolver.hpp"
#include <cmath>

double checkGradient(ImplicitODESolver& m, int accuracy = 0) {
    // accuracy can be 0, 1, 2, 3
    const double eps = 1e-3;
    static const std::array<std::vector<double>, 4> coeff =
    { { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} } };
    static const std::array<std::vector<double>, 4> coeff2 =
    { { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} } };
    static const std::array<double, 4> dd = {2, 12, 60, 840};

    VectorD xx = m.x0;
    VectorD x0b = m.x0;
    VectorD grad(m.x0.size(), 0);
    VectorD grad2(m.x0.size(), 0);
    m.computeOptimizeGradient(m.x0, grad2);

    const int innerSteps = 2*(accuracy+1);
    const double ddVal = dd[accuracy]*eps;

    for (unsigned d = 0; d < m.x0.size(); d++) {
        grad[d] = 0;
        for (int s = 0; s < innerSteps; ++s)
        {
            double tmp = xx[d];
            xx[d] += coeff2[accuracy][s]*eps;
            grad[d] += coeff[accuracy][s]*m.computeOptimizeGradient(xx, m.g);
            setZero(m.g);
            xx[d] = tmp;
        }
        grad[d] /= ddVal;
    }
    auto diff = (grad - grad2);
    double err = sqrt(sum(diff*diff))/(double)m.x0.size();
    m.x0 = x0b;
    m.computeOptimizeGradient(m.x0, m.g);
    for(int j=0; j< m.x0.size(); j++) {
        printf("%12.3f", grad[j]);
    }
    printf("\n");
    for(int j=0; j< m.x0.size(); j++) {
        printf("%12.3f", grad2[j]);
    }
    printf("\n\n");
    return err;
}

double checkDifferential(ImplicitODESolver& m, int accuracy = 0) {
    double eps = 1e-3;

    // hessian.resize(x.rows(), x.rows());
    VectorD xx = m.x0;
    VectorD gb = m.g;
    VectorD x0b = m.x0;
    auto value = [&](const VectorD& x) {
        setZero(m.g);
        return m.computeOptimizeGradient(x, m.g);
    };
    std::vector<VectorD> hessian(m.x0.size(), VectorD(m.x0.size()));
    if(accuracy == 0) {
        for (unsigned i = 0; i < m.x0.size(); i++) {
            for (unsigned j = 0; j < m.x0.size(); j++) {
                double tmpi = xx[i];
                double tmpj = xx[j];

                double f4 = value(xx);
                xx[i] += eps;
                xx[j] += eps;
                double f1 = value(xx);
                xx[j] -= eps;
                double f2 = value(xx);
                xx[j] += eps;
                xx[i] -= eps;
                double f3 = value(xx);
                hessian[i][j] = (f1 - f2 - f3 + f4) / (eps * eps);

                xx[i] = tmpi;
                xx[j] = tmpj;
            }
        }
    } else 
        for (unsigned i = 0; i < m.x0.size(); i++) {
            for (unsigned j = 0; j < m.x0.size(); j++) {
                double tmpi = xx[i];
                double tmpj = xx[j];

                double term_1 = 0;
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += -2*eps;  term_1 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += -1*eps;  term_1 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += 1*eps;   term_1 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += 2*eps;   term_1 += value(xx);

                double term_2 = 0;
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += -2*eps;  term_2 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += -1*eps;  term_2 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += 2*eps;   term_2 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += 1*eps;   term_2 += value(xx);

                double term_3 = 0;
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += -2*eps;  term_3 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += 2*eps;   term_3 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += -2*eps;  term_3 -= value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += 2*eps;   term_3 -= value(xx);

                double term_4 = 0;
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += -1*eps;  term_4 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += 1*eps;   term_4 += value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += -1*eps;  term_4 -= value(xx);
                xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += 1*eps;   term_4 -= value(xx);

                xx[i] = tmpi;
                xx[j] = tmpj;

                hessian[i][j] = (-63 * term_1+63 * term_2+44 * term_3+74 * term_4)/(600.0 * eps * eps);
            }
        
    }

    std::vector<VectorD> analyticHessian(m.x0.size(), VectorD(m.x0.size()));
    VectorD tmp(xx.size());
    double err = 0;
    for(unsigned i=0; i < xx.size(); i++) {
        for(unsigned j=0; j < xx.size(); j++) {
            VectorD a(xx.size(), 0.0);
            a[j] = 1.0;
            VectorD b(xx.size(), 0.0);
            b[i] = 1.0;
            setZero(tmp);
            m.computeOptimizeDifferential(a, tmp);
            analyticHessian[i][j] = sum(tmp*b);
            err = std::max(err, std::abs(analyticHessian[i][j] - hessian[i][j]));
        }
    }
    for(int i=0; i< m.x0.size(); i++) {
        for(int j=0; j< m.x0.size(); j++) {
            printf("%12.3f", hessian[i][j]);
        }
        printf("\n");
        for(int j=0; j< m.x0.size(); j++) {
            printf("%12.3f", analyticHessian[i][j]);
        }
        printf("\n\n");
    }
    return err;
}