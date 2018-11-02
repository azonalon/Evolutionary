#pragma once
#include <cppoptlib/problem.h>
#include <cppoptlib/solver/bfgssolver.h>
#include <Eigen/Dense>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "util/Assertions.hpp"
#include "util/Math.hpp"

#define EVOLUTIONARY_DEBUG true
#include "util/Debug.hpp"
static constexpr bool DEBUG_CHECK_DERIVATIVES = false;

class ImplicitODESolver {
 public:
  class ImplicitEulerStep : public cppoptlib::Problem<double> {
   public:
    ImplicitODESolver& s;
    ImplicitEulerStep(ImplicitODESolver& s) : s(s){};
    double value(const Eigen::VectorXd& x) override {
      s.g = 0;
      return s.computeOptimizeGradient(x, s.g);
    }
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& dest) override {
      Eigen::ArrayXd a(dest.size());
      a = 0;
      s.computeOptimizeGradient(x, a);
      dest = a;
    }
    void hessian(const Eigen::VectorXd& x, Eigen::MatrixXd& dest) override {
      Eigen::ArrayXd a(x.size());
      for (unsigned i = 0; i < dest.rows(); i++) {
        a = 0;
        a[i] = 1;
        s.g = 0;
        s.computeOptimizeDifferential(a, s.g);
        for (unsigned j = 0; j < dest.cols(); j++) {
          dest(i, j) = s.g(j);
        }
      }
    }
  };
  std::function<void(ImplicitODESolver*, double)> lineSearchHook =
      [](auto s, double alpha) { return; };
  virtual double computeGradient(const Eigen::ArrayXd& x,
                                 Eigen::ArrayXd& dest) = 0;
  virtual void precomputeStep(const Eigen::ArrayXd& x) = 0;
  virtual void computeDifferential(const Eigen::ArrayXd& x,
                                   const Eigen::ArrayXd& dx,
                                   Eigen::ArrayXd& dest) = 0;
  unsigned counter = 0;
  // virtual void timeStepFinished()=0;
  double dampPotential;
  int iNewton = 0;

  Eigen::ArrayXd dn, g, x0, x1, x2, v, xHat, temp1, temp2, M, MI, fExt, xAlpha,
      gConst, r, p, fr;
  double dG, dPhi, dX, phi, dN;
  double kDamp = 0.0;
  double dt = 0.1;
  double newtonAccuracy = 1e-3;

  void conjugateGradientSolve(
      const Eigen::ArrayXd& rhs, const Eigen::ArrayXd& initialGuess,
      std::function<void(const Eigen::ArrayXd&, Eigen::ArrayXd&)> computeLhs,
      Eigen::ArrayXd& result) {
    computeLhs(initialGuess, temp1);  // temp1=Ax0
    r = rhs - temp1;                  // temp2 = p0 = r0 = b - Ax0
    p = r;
    double rr1, rr2, alpha, beta;
    computeLhs(p, temp1);  // temp1 = Ap
    rr1 = (r * r).sum();
    alpha = rr1 / ((p * temp1).sum());
    result = initialGuess + p * alpha;  // result = x1 = x0 + alpha p0

    DMESSAGE("Conjugate Gradient Start.\n");
    for (int k = 0; k < rhs.size(); k++) {
      // if(alpha < 0) {
      //     DMESSAGE("Indefiniteness detected.");
      //     DMESSAGE("Conjugate gradient stop.");
      //     break;
      // }
      r = r - alpha * temp1;
      rr2 = (r * r).sum();
      // DERROR("Conjugate Gradient: k=%d, ||r||=%g\n", k, rr2);
      if (rr2 < 1e-5) {
        DERROR("Conjugate Gradient: k=%d, ||r||=%g\n", k, rr2);
        DMESSAGE("Conjugate gradient stop.\n");
        break;
      }
      beta = rr2 / rr1;
      rr1 = rr2;
      p = r + beta * p;
      computeLhs(p, temp1);
      alpha = rr1 / ((p * temp1).sum());
      result = result + p * alpha;
    }
  }
  void computeOptimizeDifferential(const Eigen::ArrayXd& dx,
                                   Eigen::ArrayXd& df) {
    temp2 = 0;
    computeDifferential(x0, dx, df);
    computeDifferential(x1, dx, temp2);
    df += kDamp / dt * temp2 + M * dx / dt / dt;
    // temp2 = M * dx;
    // df += M * dx / dt / dt;
  }

  void computeNewtonDirection(const Eigen::ArrayXd& g, Eigen::ArrayXd& dn) {
    // conjugateGradientSolve(g, g, [&](const auto& dx, auto& lhs) -> void {
    //     lhs = 0;
    //     temp2 = 0;
    //     computeForceDifferential(x0, dx, lhs);
    //     computeForceDifferential(x1, dx, temp2);
    //     lhs = lhs - kDamp/dt*temp2;
    //     temp2 = M*dx;
    //     lhs += temp2/dt/dt;
    // }, dn);
    conjugateGradientSolve(g, g,
                           [&](const auto& dx, auto& df) {
                             df = 0;
                             computeOptimizeDifferential(dx, df);
                           },
                           dn);
    dn = -dn;
  };

  void computeDPhi(const Eigen::ArrayXd& g, const Eigen::ArrayXd& dn) {
    dPhi = (g * dn).sum();
  }

  double computeOptimizeGradient(const Eigen::ArrayXd& x,
                                 Eigen::ArrayXd& dest) {
    double energy = 0;

    v = (x - x1) / dt;
    temp2 = 0;
    computeDifferential(x1, v, temp2);
    dampPotential = (v * temp2).sum() * kDamp / 2 * dt;
    energy += dampPotential;
    dest = kDamp * temp2;

    energy += computeGradient(x, dest);

    temp1 = x - xHat;
    temp2 = M * temp1;
    energy += (temp1 * temp2).sum() / 2.0 / dt / dt;
    dest += temp2 / (dt * dt);

    return energy;
  }

  ImplicitODESolver(unsigned n, const std::vector<double>& masses)
      : dn(n), g(n), x0(n), x1(n), x2(n), v(n), xHat(n), temp1(n), temp2(n),
        M(n), MI(n), fExt(n), xAlpha(n), gConst(n), r(n), p(n), fr(n)
  // Eigen::ArrayXd dn, g, x0, x1, x2, v, xHat,
  //             temp1, temp2, M, MI, fExt,
  //             xAlpha, gConst, r, p;
  {
    assert(n == masses.size());

    dn = 0; g = 0; x0 = 0; x1 = 0; x2 = 0; v = 0; xHat = 0; temp1 = 0; temp2 = 0;
    M = 1; MI = 1; fExt = 0; fr = 0; xAlpha = 0; gConst = 0; r = 0; p = 0;
    int i = 0;
    for (double m : masses) {
      assert(M[i] > 0);
      M[i] = m;
      MI[i] = 1 / m;
      i++;
    }
  }

  double newtonStep() {
    // double k = 1e-2;
    double k = 1e-2;
    double alpha = 1;
    double alphaMax = 1e3;
    double l = 1e3;

    g = 0;
    phi = computeOptimizeGradient(x0, g);
    dG = (g * g).sum();
    if (dG < newtonAccuracy) {
      return dG;
    }
    computeNewtonDirection(g, dn);

    dN = (dn * dn).sum();
    if(dN < 1e-7) {
      return 0.0;
    }
    computeDPhi(g, dn);
    // printf("Phi=%g, Dphi=%g\n, dN=%g\n", phi, dPhi, dN);

    double dPhiSq = dPhi * std::abs(dPhi);
    if (dPhiSq < -k * dN * dG) {
      // dN is suitable
      DERROR("dn is suitable! dPhi*abs(dPhi)=%g, dN*dG=%g\n", dPhiSq, dN * dG);
      // log.printf(DEBUg, "dn is suitable! dNdG=%g, dn dg=%g\n", dNdG, dN*dg);
    } else if (dPhiSq > k * dN * dG) {
      // -dN is suitable
      DERROR("-dn is suitable! dPhi*abs(dPhi)=%g, dN*dG=%g\n", dPhiSq, dN * dG);
      dPhi = -dPhi;
      dn = -dn;
    } else {
      DERROR("gradient is suitable! dNdG=%g, dN*dG=%g\n", dPhi * dPhi, dN * dG);
      dn = -g;
      dPhi =-dG;
      dN = dG;
      alpha = 0.01*alpha;
      // dn.set(g);
    }
    // if (dN > l) {
    //   dn = dn * l / dN;
    //   dN = l;
    // }
    alpha = strongWolfeLineSearch(alpha, alphaMax);
    lineSearchHook(this, alpha);
    x0 = x0 + alpha * dn;
    dG = (g * g).sum();
    // return dG * (1 - alpha) * 2;
    return dG;
  }

  double strongWolfeLineSearch(double alpha, double alphaMax) {
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
    // scanLineToFile(5*alpha, 100, String.format("scanline_%04.4f_%02d.dat", t,
    // iNewton));
    int j = 1;
    // lineSearchStep(alpha);
    DMESSAGE("Line search start.");
    while (j < 30) {
      computePhiDPhi(alpha1);
      phi1 = phi;
      dPhi1 = dPhi;
      // if(std::abs(phi) < 1e-2 && std::abs(dPhi) < 1e-2) return alpha1;
      // if(std::abs(dPhi) < 1e-2) return alpha1;
      if (std::abs(dPhi1) <= -c2 * dPhiS) {
        DERROR(
            "Case phi is small enough and dPhi is small enough:\n    phiS=%g, "
            "phi=%g, dPhiS=%g, dPhi=%g\n",
            phiS, phi1, dPhiS, dPhi1);
        DERROR("LineSearch returned with alpha=%g\n", alpha1);
        return alpha1;
      }
      if (phi1 > phiS + c1 * alpha1 * dPhiS || (j > 1 && phi1 >= phi0)) {
        DERROR("Case phi is too big: phiS=%g, dPhi=%g, phiWolfe=%g\n", phiS,
               dPhiS, phiS + c1 * alpha * dPhiS);
        alpha1 = zoom(alpha0, phi0, dPhi0, alpha1, phi1, dPhi1, phiS, dPhiS, c1,
                      c2, alpha0, phi0, dPhi0, alpha1, phi1, dPhi1);
        DERROR("Zoom returned with alpha=%g, phi=%g, dphi=%g\n", alpha1, phi,
               dPhi);
        return alpha1;
      }
      if (dPhi1 >= 0) {
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
      alpha1 = choose(alpha1, phi, dPhi, alphaMax);
      phi1 = phi;
      dPhi1 = dPhi;
      j++;
    }
    DMESSAGE("Line search end.");
    throw std::runtime_error("Line Search did not end");
    return 1;
  }

  inline double zoom(double lo, double philo, double dPhilo, double hi,
                     double phihi, double dPhihi, double phiS, double dPhiS,
                     double c1, double c2, double alpha0, double phi0,
                     double dPhi0, double alpha1, double phi1, double dPhi1) {
    int j = 0;
    double alpha=1;
    double e = 1e-6;
    DMESSAGE("Zoom start.");
    while (j < 5) {
      alpha = interpolate(lo, philo, dPhilo, hi, phihi, dPhihi);
      DERROR(
          "Interp: \nlo=%g, philo=%g, dlo=%g,\nhi=%g, phihi=%g, dhi=%g, alpha=%g\n",
          lo, philo, dPhilo, hi, phihi, dPhihi, alpha);
      assert(std::isfinite(alpha));
      computePhiDPhi(alpha);
      if (phi > phiS + c1 * alpha * dPhiS || (phi >= philo)) {
        hi = alpha;
        phihi = phi;
        dPhihi = dPhi;
      } else {
        if (std::abs(dPhi) <= -c2 * dPhiS) {
          return alpha;
        }
        if (dPhi * (hi - lo) >= 0) {
          hi = lo;
          phihi = philo;
          dPhihi = dPhilo;
        }
        lo = alpha;
        philo = phi;
        dPhilo = dPhi;
      }
      if (std::abs(alpha - alpha1) < e) {
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

  static double interpolate(double a, double fa, double dfa, double b,
                            double fb, double dfb) {
    double c = 5e-2;
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
    assert(x <= std::max(a, b) && x >= std::min(a, b) && "interp alpha not in interval" );
    assert(std::isfinite(x));
    return x;
  }

  static double choose(double alpha1, double phi, double dphi,
                       double alphaMax) {
    // TODO: make it smart
    std::cout << "choose was called\n"; 
    return std::min(2 * alpha1, alphaMax);
  }

  void computePhiDPhi(double alpha) {
    xAlpha = alpha * dn + x0;
    g = 0;
    phi = computeOptimizeGradient(xAlpha, g);
    // dPhi = dot(dn, g)/normP2(dn);
    computeDPhi(g, dn);
  }

  void computeForwardEulerStep(Eigen::ArrayXd& x2, Eigen::ArrayXd& x1,
                               Eigen::ArrayXd& x0, const Eigen::ArrayXd& fExt) {
    // assert(counter == 0 || v.isApprox((x0-x1)/dt, 1e-24));
    counter++;
    x2 = x1;
    x1 = x0;
    x0 = dt * v + x0;
    temp1 = fExt * MI;
    x0 = dt * dt * temp1 + x0;
    // temp1 =  (x1 - x2)/dt;
    // DERROR("||v||=%g\n", sqrt((temp1*temp1).sum()));
  }

  void implicitEulerStep() {
    double dG = 1e33;
    double phiOld = std::numeric_limits<double>::max();

    DERROR("Newton iteration start. dG=%g\n",
           dG);  // std::cout << "x2=" <<  x2 << std::endl;
    // step forward and set the initial guess

    // compute initital guess
    computeForwardEulerStep(x2, x1, x0, fExt);
    xHat = x0;
    precomputeStep(x0);

    if (DEBUG_CHECK_DERIVATIVES) {
      ImplicitEulerStep step(*this);
      // cppoptlib::BfgsSolver<ImplicitEulerStep> solver;
      // Eigen::VectorXd x0v = x0;
      g = 0;
      double e = computeOptimizeGradient(x0, g);
      // std::cout << "energy " << e << std::endl;
      // std::cout << "gradient " << g.transpose() << std::endl;
      assert(step.checkGradient(x0));
      assert(step.checkHessian(x0));
      // solver.minimize(step, x0v);
      // x0 = x0v;
      // return;
    }

    iNewton = 0;
    while (iNewton <= 40) {
      dG = newtonStep();
      if (dG < newtonAccuracy) {
        DERROR("Newton iteration stopped: i=%d, dG=%g\n", iNewton, dG);
        return;
      }
      if (phi >= phiOld * 1.2) {
        DERROR("Energy minimization did not converge!\n phi=%g, phiOld=%g\n",
                phi, phiOld);
        // assert(false);
        // break;
        return;
      }
      phiOld = phi;
      iNewton++;
      DERROR("Newton iteration i=%d, dG=%g, phi=%g\n",  iNewton, dG, phi);
    }
    // char message[200];
    // sprintf(message,
    //         "Energy minimization did not stop after 40 iterations! \
    //                        dE=%g, dE'=%g\n\n",
    //         dG, dG);
    // throw std::runtime_error(message);
  }
};
