#include "util/Math.hpp"

#ifndef EVOLUTIONARY_DEBUG
#define EVOLUTIONARY_DEBUG false
#endif
static constexpr bool DEBUG_CHECK_DERIVATIVES = false;
static constexpr bool DEBUG_LINESEARCH = false;
static constexpr bool SKIP_LINESEARCH = true;
#pragma once

// TODO: Implement swapping of pointers for the arrays instead of copying

class ImplicitODESolver
{
public:
    std::function<void(ImplicitODESolver *, double)> lineSearchHook =
        [](auto s, double alpha) { return; };
    unsigned counter = 0;
    // virtual void timeStepFinished()=0;
    double dampPotential;
    int iNewton = 0;
    VectorD dn, g, x0, x1, x2, v, xHat, temp1, temp2, M, MI, fExt, xAlpha,
        r, p, fr;
    // modelForces for rendering arrows representing force
    VectorD modelForces;
    double dG, dPhi, dX, phi, dN;
    double kDamp = 0.0;
    double dt = 10;
    double newtonAccuracy = 1e-3;
    unsigned dim;

    void computeOptimizeDifferential(const VectorD &dx,
                                     VectorD &df);

    double computeOptimizeGradient(const VectorD &x,
                                   VectorD &dest);
    void implicitEulerStep(void);

    ImplicitODESolver(unsigned n, const std::vector<double> &masses);

private:
    void conjugateGradientSolve(
        const VectorD &rhs, const VectorD &initialGuess,
        std::function<void(const VectorD &, VectorD &)> computeLhs,
        VectorD &result);
    void computeNewtonDirection(const VectorD &g, VectorD &dn);
    void computeDPhi(const VectorD &g, const VectorD &dn);
    void computePhiDPhi(double alpha);
    double newtonStep();
    double strongWolfeLineSearch(double alpha, double alphaMax);
    double zoom(double lo, double philo, double dPhilo, double hi,
                    double phihi, double dPhihi, double phiS, double dPhiS,
                    double c1, double c2, double alpha0, double phi0,
                    double dPhi0, double alpha1, double phi1, double dPhi1);
    virtual double computeGradient(const VectorD &x,
                                   VectorD &dest) = 0;
    virtual void precomputeStep(const VectorD &x) = 0;
    virtual void computeDifferential(const VectorD &x,
                                     const VectorD &dx,
                                     VectorD &dest) = 0;
    void computeForwardEulerStep(VectorD &x2, VectorD &x1,
                                                    VectorD &x0,
                                                    const VectorD &fExt);
};
