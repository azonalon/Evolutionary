#include "util/Debug.hpp"
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
  while (j < 5) {
    computePhiDPhi(alpha1);
    phi1 = phi;
    dPhi1 = dPhi;
    if (std::abs(phi) < 1e-2 && std::abs(dPhi) < 1e-2) return alpha1;
    if (phi1 > phiS + c1 * alpha1 * dPhiS || (j > 1 && phi1 >= phi0)) {
      DERROR("Case phi is too big: phiS=%g, dPhi=%g, phiWolfe=%g\n", phiS,
             dPhiS, phiS + c1 * alpha * dPhiS);
      alpha1 = zoom(alpha0, phi0, dPhi0, alpha1, phi1, dPhi1, phiS, dPhiS, c1,
                    c2, alpha0, phi0, dPhi0, alpha1, phi1, dPhi1);
      DERROR("Zoom returned with alpha=%g, phi=%g, dphi=%g\n", alpha1, phi,
             dPhi);
      return alpha1;
    }
    if (std::abs(dPhi1) <= -c2 * dPhiS) {
      DERROR(
          "Case phi is small enough and dPhi is small enough:\n    phiS=%g, "
          "phi=%g, dPhiS=%g, dPhi=%g\n",
          phiS, phi1, dPhiS, dPhi1);
      DERROR("LineSearch returned with alpha=%g\n", alpha1);
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
  double alpha;
  double e = 1e-6;
  DMESSAGE("Zoom start.");
  while (j < 30) {
    alpha = interpolate(lo, philo, dPhilo, hi, phihi, dPhihi);
    DERROR(
        "Interp: lo=%g, philo=%g, dlo=%g,\n \
                                      hi=%g, phihi=%g, dhi=%g, alpha=%g\n",
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
  throw std::runtime_error("Zoom did not converge");
  return -1;
}

static double interpolate(double a, double fa, double dfa, double b, double fb,
                          double dfb) {
  double c = 5e-2;
  double d = b - a;
  double s = Math::signum(d);
  double d1 = dfa + dfb - 3 * (fa - fb) / (a - b);
  double k = d1 * d1 - dfa * dfb;
  double d2 = s * std::sqrt(k);
  double x = b - d * ((dfb + d2 - d1) / (dfb - dfa + 2 * d2));
  if (std::abs(a - x) < c * d * s) {
    DERROR("Halving Interval: a=%g x=%g b=%g", a, x, b);
    x = a + 2 * c * d;
  }
  if (std::abs(b - x) < c * d * s) {
    DERROR("Halving Interval: a=%g x=%g b=%g", a, x, b);
    x = b + 2 * c * d;
  }
  assertTrue("Interpolated alpha is not in interval.",
             x <= std::max(a, b) && x >= std::min(a, b));
  return x;
}

static double choose(double alpha1, double phi, double dphi, double alphaMax) {
  // TODO: make it smart
  return std::min(2 * alpha1, alphaMax);
}