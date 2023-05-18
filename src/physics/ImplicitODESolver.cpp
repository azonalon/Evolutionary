#include "physics/ImplicitODESolver.hpp"
#include "util/CheckDerivatives.hpp"
#include "util/Debug.hpp"

static inline double interpolate(double a, double fa, double dfa, double b, double fb,
                          double dfb)
{
    // double c = 5e-2;
    double d = b - a;
    double s = Math::signum(d);
    double d1 = dfa + dfb - 3 * (fa - fb) / (a - b);
    double k = d1 * d1 - dfa * dfb;
    double d2 = s * std::sqrt(k);
    double x = b - d * ((dfb + d2 - d1) / (dfb - dfa + 2 * d2));
    // if (std::abs(a - x) < c * d * s) {
    //   DERROR("Halving Interval: a=%g x=%g b=%g", a, x, b);
    //   x = a + 2 * c * d;
    // }
    // if (std::abs(b - x) < c * d * s) {
    //   DERROR("Halving Interval: a=%g x=%g b=%g", a, x, b);
    //   x = b + 2 * c * d;
    // }
    assert(x <= std::max(a, b) && x >= std::min(a, b) &&
           "interp alpha not in interval");
    assert(std::isfinite(x));
    return x;
}


double ImplicitODESolver::zoom(double lo, double philo, double dPhilo, double hi,
                   double phihi, double dPhihi, double phiS, double dPhiS,
                   double c1, double c2, double alpha0, double phi0,
                   double dPhi0, double alpha1, double phi1, double dPhi1)
{
    int j = 0;
    double alpha = 1;
    double e = 1e-6;
    DMESSAGE("Zoom start.");
    while (j < 5)
    {
        alpha = interpolate(lo, philo, dPhilo, hi, phihi, dPhihi);
        DERROR(
            "Interp: \nlo=%g, philo=%g, dlo=%g,\nhi=%g, phihi=%g, dhi=%g, "
            "alpha=%g\n",
            lo, philo, dPhilo, hi, phihi, dPhihi, alpha);
        assert(std::isfinite(alpha));
        computePhiDPhi(alpha);
        if (phi > phiS + c1 * alpha * dPhiS || (phi >= philo))
        {
            hi = alpha;
            phihi = phi;
            dPhihi = dPhi;
        }
        else
        {
            if (std::abs(dPhi) <= -c2 * dPhiS)
            {
                return alpha;
            }
            if (dPhi * (hi - lo) >= 0)
            {
                hi = lo;
                phihi = philo;
                dPhihi = dPhilo;
            }
            lo = alpha;
            philo = phi;
            dPhilo = dPhi;
        }
        if (std::abs(alpha - alpha1) < e)
        {
            return alpha;
        }
        alpha0 = alpha1;
        phi0 = phi1;
        dPhi0 = dPhi1;
        phi1 = phi;
        dPhi1 = dPhi;
        alpha1 = alpha;
        j++;
    }
    return alpha;
    DERROR("Zoom did not converge, alpha=%g", alpha);
    // return -1;
}

static inline double choose(double alpha1, double phi, double dphi,
                                        double alphaMax)
{
    // TODO: make it smart
    double result = std::min(2 * alpha1, alphaMax);
    printf("choose called with alphamax=%g, alpha1=%g returned alpha=%g\n",
           alphaMax, alpha1, result);
    return result;
}


inline void ImplicitODESolver::conjugateGradientSolve(
    const VectorD &rhs, const VectorD &initialGuess,
    std::function<void(const VectorD &, VectorD &)> computeLhs, VectorD &result)
{
    computeLhs(initialGuess, temp1); // temp1=Ax0
    r = rhs - temp1;                 // temp2 = p0 = r0 = b - Ax0
    p = r;
    double rr1, rr2, alpha, beta;
    computeLhs(p, temp1); // temp1 = Ap
    rr1 = sum(r * r);
    alpha = rr1 / sum(p * temp1);
    result = initialGuess + p * alpha; // result = x1 = x0 + alpha p0

    DMESSAGE("Conjugate Gradient Start.\n");
    for (unsigned k = 0; k < rhs.size(); k++)
    {
        // if(alpha < 0) {
        //     DMESSAGE("Indefiniteness detected.");
        //     DMESSAGE("Conjugate gradient stop.");
        //     break;
        // }
        r = r - temp1 * alpha;
        rr2 = sum(r * r);
        // DERROR("Conjugate Gradient: k=%d, ||r||=%g\n", k, rr2);
        if (rr2 < 1e-5)
        {
            DERROR("Conjugate Gradient: k=%d, ||r||=%g\n", k, rr2);
            DMESSAGE("Conjugate gradient stop.\n");
            break;
        }
        beta = rr2 / rr1;
        rr1 = rr2;
        p = r + p * beta;
        computeLhs(p, temp1);
        alpha = rr1 / sum(p * temp1);
        result = result + p * alpha;
    }
}
void ImplicitODESolver::computeOptimizeDifferential(const VectorD &dx,
                                                    VectorD &df)
{
    setZero(temp2);
    computeDifferential(x0, dx, df);
    computeDifferential(x1, dx, temp2);
    df += temp2 * (kDamp / dt) + M * dx * (1.0 / dt / dt);
}
double ImplicitODESolver::computeOptimizeGradient(const VectorD &x,
                                                  VectorD &dest)
{
    double energy = 0.0;

    v = (x - x1) * (1 / dt);
    setZero(temp2);
    computeDifferential(x1, v, temp2);
    dampPotential = sum(v * temp2) * (kDamp / 2 * dt);
    energy += dampPotential;
    dest = temp2 * kDamp;

    energy += computeGradient(x, dest);
    dest += fExt * MI;
    energy += sum(fExt * x * MI);
    modelForces = dest;

    temp1 = x - xHat;
    temp2 = M * temp1;
    energy += sum(temp1 * temp2) / 2.0 / dt / dt;
    dest += temp2 * (1 / (dt * dt));

    return energy;
}

inline void ImplicitODESolver::computeNewtonDirection(const VectorD &g, VectorD &dn)
{
    conjugateGradientSolve(
        g, g,
        [&](const auto &dx, auto &df) {
            setZero(df);
            computeOptimizeDifferential(dx, df);
        },
        dn);
    negate(dn);
};

inline void ImplicitODESolver::computeDPhi(const VectorD &g, const VectorD &dn)
{
    dPhi = sum(g * dn);
}


ImplicitODESolver::ImplicitODESolver(unsigned n,
                                     const std::vector<double> &masses)
    : dn(n, 0.0),
      g(n, 0.0),
      x0(n, 0.0),
      x1(n, 0.0),
      x2(n, 0.0),
      v(n, 0.0),
      xHat(n, 0.0),
      temp1(n, 0.0),
      temp2(n, 0.0),
      M(n, 1.0),
      MI(n, 1.0),
      fExt(n, 0.0),
      xAlpha(n, 0.0),
      r(n, 0.0),
      p(n, 0.0),
      fr(n, 0.0),
      modelForces(n, 0.0),
      dim(n)
{
    assert(n == masses.size());
    int i = 0;
    for (double m : masses)
    {
        assert(M[i] > 0);
        M[i] = m;
        MI[i] = 1 / m;
        i++;
    }
}

inline double ImplicitODESolver::newtonStep()
{
    double k = 1e-2;
    double alpha = 1;
    double alphaMax = 1e3;
    double l = 1e3;

    setZero(g);
    phi = computeOptimizeGradient(x0, g);
    dG = sum(g * g);
    if (DEBUG_CHECK_DERIVATIVES)
    {
        // checkDifferential(*this, 0);
        // double err = checkGradient(*this);
    }

    if (dG < newtonAccuracy / dim)
    {
        return dG;
    }

    computeNewtonDirection(g, dn);
    dN = sum(dn * dn);

    if (SKIP_LINESEARCH)
    {
        x0 = x0 + dn * alpha;
        // phi = computeOptimizeGradient(x0, g);
        // dG = (g * g).sum();
        return dN;
    }

    computeDPhi(g, dn);
    // printf("Phi=%g, Dphi=%g\n, dN=%g\n", phi, dPhi, dN);

    double dPhiSq = dPhi * std::abs(dPhi);
    if (dPhiSq < -k * dN * dG)
    {
        // dN is suitable
        DERROR("dn is suitable! dPhi*abs(dPhi)=%g, dN*dG=%g\n", dPhiSq, dN * dG);
        // log.printf(DEBUg, "dn is suitable! dNdG=%g, dn dg=%g\n", dNdG, dN*dg);
    }
    else if (dPhiSq > k * dN * dG)
    {
        // -dN is suitable
        DERROR("-dn is suitable! dPhi*abs(dPhi)=%g, dN*dG=%g\n", dPhiSq, dN * dG);
        dPhi = -dPhi;
        negate(dn);
    }
    else
    {
        DERROR("gradient is suitable! dNdG=%g, dN*dG=%g\n", dPhi * dPhi, dN * dG);
        dn = g * (-1.0);
        dPhi = dG * (-1.0);
        dN = dG;
        alpha = 0.01 * alpha;
    }

    alpha = strongWolfeLineSearch(alpha, alphaMax);
    x0 = x0 + dn * alpha;
    phi = computeOptimizeGradient(x0, g);
    dG = sum(g * g);
    // return dG * (1 - alpha) * 2;
    return dG;
}

inline double ImplicitODESolver::strongWolfeLineSearch(double alpha, double alphaMax)
{
    double phiS = phi;
    double dPhiS = dPhi;
    double alpha0 = 0;
    double alpha1 = alpha;
    double phi0 = phi;
    double dPhi0 = dPhi;
    double phi1 = phi;
    double dPhi1 = dPhi;
    double c1 = 1e-2;
    double c2 = 1e-2;
    int j = 1;
    DMESSAGE("Line search start.");
    while (j < 30)
    {
        computePhiDPhi(alpha1);
        phi1 = phi;
        dPhi1 = dPhi;
        // if(std::abs(phi) < 1e-2 && std::abs(dPhi) < 1e-2) return alpha1;
        // if(std::abs(dPhi) < 1e-2) return alpha1;
        if (std::abs(dPhi1) <= -c2 * dPhiS)
        {
            DERROR(
                "Case phi is small enough and dPhi is small enough:\n    phiS=%g, "
                "phi=%g, dPhiS=%g, dPhi=%g\n",
                phiS, phi1, dPhiS, dPhi1);
            DERROR("LineSearch returned with alpha=%g\n", alpha1);
            return alpha1;
        }
        if (phi1 > phiS + c1 * alpha1 * dPhiS || (j > 1 && phi1 >= phi0))
        {
            DERROR("Case phi is too big: phiS=%g, dPhi=%g, phiWolfe=%g\n", phiS,
                   dPhiS, phiS + c1 * alpha * dPhiS);
            alpha1 = zoom(alpha0, phi0, dPhi0, alpha1, phi1, dPhi1, phiS, dPhiS, c1,
                          c2, alpha0, phi0, dPhi0, alpha1, phi1, dPhi1);
            DERROR("Zoom returned with alpha=%g, phi=%g, dphi=%g\n", alpha1, phi,
                   dPhi);
            return alpha1;
        }
        if (dPhi1 >= 0)
        {
            DERROR(
                "Case phi is small and dPhi is big and positive:\n    phiS=%g, "
                "dPhiS=%g, phi=%g, dPhi=%g\n",
                phiS, dPhiS, phi, dPhi);
            alpha1 = zoom(alpha1, phi1, dPhi1, alpha0, phi0, dPhi0, phiS, dPhiS, c1,
                          c2, alpha0, phi0, dPhi0, alpha1, phi1, dPhi1);
            DERROR("Zoom returned with alpha=%g, phi=%g, dphi=%g\n", alpha1, phi,
                   dPhi);
            return alpha1;
        }
        DERROR(
            "Case phi is small and dPhi is big and negative:\n    dPhiS=%g, "
            "dPhi=%g\n",
            dPhiS, dPhi);
        alpha0 = alpha1;
        phi0 = phi1;
        dPhi0 = dPhi1;
        lineSearchHook(this, alpha1);
        alpha1 = choose(alpha1, phi, dPhi, alphaMax);
        if (DEBUG_LINESEARCH)
        {
            // lineSearchDebugDumpFile(alpha1, -2, 2);
        }
        phi1 = phi;
        dPhi1 = dPhi;
        j++;
    }
    DMESSAGE("Line search end.");
    throw std::runtime_error("Line Search did not end");
    return 1;
}

inline void ImplicitODESolver::computePhiDPhi(double alpha)
{
    xAlpha = dn * alpha + x0;
    setZero(g);
    phi = computeOptimizeGradient(xAlpha, g);
    // dPhi = dot(dn, g)/normP2(dn);
    computeDPhi(g, dn);
}

inline void ImplicitODESolver::computeForwardEulerStep(VectorD &x2, VectorD &x1,
                                                VectorD &x0,
                                                const VectorD &fExt)
{
    // assert(counter == 0 || v.isApprox((x0-x1)/dt, 1e-24));
    counter++;
    // x2 = x1;
    x1 = x0;
    x0 += v * dt;
    // v = (x0 - x1)/dt;
    // temp1 = fExt * MI;
    // x0 = dt * dt * temp1 + x0;
    // temp1 =  (x1 - x2)/dt;
    // DERROR("||v||=%g\n", sqrt((temp1*temp1).sum()));
}

void ImplicitODESolver::implicitEulerStep(void)
{
    double dG = 1e33;
    double phiOld = std::numeric_limits<double>::max();

    DERROR("Newton iteration start. dG=%g\n",
           dG); // std::cout << "x2=" <<  x2 << std::endl;
    // step forward and set the initial guess

    // compute initital guess
    computeForwardEulerStep(x2, x1, x0, fExt);
    xHat = x0;
    precomputeStep(x0);


    iNewton = 0;
    while (iNewton <= 10)
    {
        dG = newtonStep();
        // if (dG < newtonAccuracy/dim) {
        //   DERROR("Newton iteration stopped: i=%d,
        //   dG=%g\n---------------------------------\n", iNewton, dG); return;
        // }
        // if (phi >= phiOld) {
        //   DERROR("Energy minimization did not converge!\n phi=%g, phiOld=%g\n",
        //           phi, phiOld);
        //   // assert(false);
        //   // break;
        //   return;
        // }
        phiOld = phi;
        iNewton++;
        DERROR("Newton iteration i=%d, dG=%g, phi=%g\n", iNewton, dG, phi);
    }
    DERROR(
        "Newton iteration stopped: i=%d, "
        "dG=%g\n---------------------------------\n",
        iNewton, dG);
    // char message[200];
    // sprintf(message,
    //         "Energy minimization did not stop after 40 iterations!
    //                        dE=%g, dE'=%g\n\n",
    //         dG, dG);
    // throw std::runtime_error(message);
}
