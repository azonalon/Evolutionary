#pragma once
#include <Eigen/Dense>
#include "util/Debug.hpp"
#include "util/Math.hpp"
#include "util/Assertions.hpp"
#include <cmath>
#include <stdexcept>
#include <climits>
#include <vector>

class ImplicitODESolver {
public:
    virtual double computeForce(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest)=0;
    virtual void computeForceDifferential(const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest)=0;
    // virtual void timeStepFinished()=0;
    double dampPotential;
    int iNewton=0;

    Eigen::ArrayXd dn, g, x0, x1, x2, v, xHat,
                temp1, temp2, M, MI, fExt,
                xAlpha, gConst, r, p;
    double dG, dPhi, dX, phi, dN;
    double kDamp=0.0;
    double dt=0.1;
    double newtonAccuracy = 1e-5;

    void conjugateGradientSolve(const Eigen::ArrayXd& rhs, const Eigen::ArrayXd& initialGuess,
                                      std::function<void(const Eigen::ArrayXd&,Eigen::ArrayXd&)>
                                        computeLhs,
                                      Eigen::ArrayXd& result) {
        computeLhs(initialGuess, temp1); // temp1=Ax0
        r = rhs - temp1; // temp2 = p0 = r0 = b - Ax0
        p = r;
        double rr1, rr2, alpha, beta;
        computeLhs(p, temp1); // temp1 = Ap
        rr1 = (r*r).sum();
        alpha=rr1/((p*temp1).sum());
        result = initialGuess + p*alpha; // result = x1 = x0 + alpha p0

        DMESSAGE("Conjugate Gradient Start.\n");
        for(int k=0; k<rhs.size(); k++) {
            // if(alpha < 0) {
            //     DMESSAGE("Indefiniteness detected.");
            //     DMESSAGE("Conjugate gradient stop.");
            //     break;
            // }
            r = r - alpha * temp1;
            rr2 = (r*r).sum();
            DERROR("Conjugate Gradient: k=%d, ||r||=%g\n", k, rr2);
            if(rr2 < 1e-5 ) {
                DMESSAGE("Conjugate gradient stop.\n");
                break;
            }
            beta = rr2/rr1;
            rr1 = rr2;
            p = r + beta*p;
            computeLhs(p, temp1);
            alpha = rr1/((p*temp1).sum());
            result = result + p*alpha;
        }
    }

    void computeNewtonDirection(const Eigen::ArrayXd& g, Eigen::ArrayXd& dn) {
        conjugateGradientSolve(g, g, [&](auto& dx, auto& lhs) -> void {
            assert((x0*x0).sum() > 0);
            assert((x1*x1).sum() > 0);
            computeForceDifferential(x0, dx, lhs);
            computeForceDifferential(x1, dx, temp2);
            lhs = lhs - kDamp/dt*temp2;
            temp2 = M*dx;
            lhs += temp2/dt/dt;
        }, dn);
        dn = -dn;
    };

    void computeDPhi(const Eigen::ArrayXd& g, const Eigen::ArrayXd& dn) {
        dPhi = (g*dn).sum();
    }

    double computeOptimizeGradient(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest) {
        double energy = computeForce(x, dest);

        temp1 = x - xHat;
        temp2 = M * temp1;
        energy += (temp1*temp2).sum()/2/dt/dt;
        dest = temp2/(dt*dt) - dest;
        v = (x - x1)/dt;
        computeForceDifferential(x1, v, temp2);
        dampPotential = -(v*temp2).sum()*kDamp/2*dt;
        energy += dampPotential;
        // DERROR("Compute Gradient: damping dampPotential=%g", dampPotential);
        dest = -kDamp*temp2 + dest;
        return energy;
    }


    ImplicitODESolver(int n, const std::vector<double>& masses) :
        dn(n), g(n),  x0(n), x1(n), x2(n), v(n), xHat(n),
        temp1(n), temp2(n),M(n), MI(n), fExt(n),xAlpha(n), gConst(n), r(n), p(n)
    // Eigen::ArrayXd dn, g, x0, x1, x2, v, xHat,
    //             temp1, temp2, M, MI, fExt,
    //             xAlpha, gConst, r, p;
    {
        assert(n == masses.size());

        dn=0; g=0; x0=0; x1=0; x2=0; v=0; xHat=0;
        temp1=0; temp2=0; M=1; MI=1; fExt=0;
        xAlpha=0; gConst=0; r=0; p=0;
        int i=0;
        for(double m: masses) {
            assert(M[i] > 0);
            M[i] = m;
            MI[i] = 1/m;
            i++;
        }
    }

    double newtonStep() {
        double k = 1e-2;
        double alpha = 1;
        double alphaMax  = 1e3;
        double l = 1e3;

        phi = computeOptimizeGradient(x0, g);
        dG = (g*g).sum();
        if(dG < newtonAccuracy) {
            return dG;
        }
        computeNewtonDirection(g, dn);

        dN = (dn*dn).sum();
        computeDPhi(g, dn);

        double dPhiSq = dPhi*std::abs(dPhi);
        if(dPhiSq < -k*dN*dG) {
            // dN is suitable
            // log.printf(DEBUg, "dn is suitable! dNdG=%g, dn dg=%g\n", dNdG, dN*dg);
        }
        else if(dPhiSq > k*dN*dG) {
            // -dN is suitable
            DERROR("-dn is suitable! dPhi*abs(dPhi)=%g, dN*dG=%g\n", dPhiSq, dN*dG);
            dPhi = -dPhi;
            dn = -dn;
        }
        else {
            DERROR("gradient is suitable! dNdG=%g, dN*dG=%g\n", dPhi*dPhi, dN*dG);
            dn = -g;
            dPhi = -dG;
            dN = dG;
            // dn.set(g);
        }
        if(dN > l) {
            dn = dn * l/dN;
            dN = l;
        }
        alpha = strongWolfeLineSearch(alpha, alphaMax);
        x0 = x0 + alpha*dn;
        dG = (g*g).sum();
        return dG;
    }

    double strongWolfeLineSearch(double alpha, double alphaMax) {
        double phiS  = phi;
        double dPhiS = dPhi;
        double alpha0 = 0;
        double alpha1 = alpha;
        double phi0 = phi;
        double dPhi0 = dPhi;
        double phi1 = phi;
        double dPhi1 = dPhi;
        double c1  = 1e-2;
        double c2  = 1e-2;
        // scanLineToFile(5*alpha, 100, String.format("scanline_%04.4f_%02d.dat", t, iNewton));
        int j = 1;
        // lineSearchStep(alpha);
        DMESSAGE("Line search start.");
        while(j<5) {
            computePhiDPhi(alpha1);
            phi1 = phi;
            dPhi1 = dPhi;
            if(std::abs(phi) < 1e-20 && std::abs(dPhi) < 1e-20) return alpha1;
            if(phi1 > phiS + c1*alpha1*dPhiS || (j>1 && phi1 >= phi0)) {
                DERROR("Case phi is too big: phiS=%g, dPhi=%g, phiWolfe=%g\n",
                                  phiS, dPhiS, phiS + c1*alpha*dPhiS);
                alpha1 = zoom(alpha0, phi0, dPhi0, alpha1, phi1, dPhi1,
                              phiS, dPhiS, c1, c2,
                              alpha0, phi0, dPhi0, alpha1, phi1, dPhi1);
                DERROR("Zoom returned with alpha=%g, phi=%g, dphi=%g\n", alpha1, phi, dPhi);
                return alpha1;
            }
            if(std::abs(dPhi1) <= -c2*dPhiS) {
                DERROR("Case phi is small enough and dPhi is small enough:\n    phiS=%g, phi=%g, dPhiS=%g, dPhi=%g\n",
                                  phiS, phi1, dPhiS, dPhi1);
                DERROR("LineSearch returned with alpha=%g\n", alpha1);
                return alpha1;
            }
            if(dPhi1 >= 0) {
                DERROR("Case phi is small and dPhi is big and positive:\n    phiS=%g, dPhiS=%g, phi=%g, dPhi=%g\n", phiS, dPhiS, phi, dPhi);
                alpha1 = zoom(alpha1, phi1, dPhi1, alpha0, phi0, dPhi0,
                              phiS, dPhiS, c1, c2,
                              alpha0, phi0, dPhi0, alpha1, phi1, dPhi1);
                DERROR("Zoom returned with alpha=%g, phi=%g, dphi=%g\n", alpha1, phi, dPhi);
                return alpha1;
            }
            DERROR("Case phi is small and dPhi is big and negative:\n    dPhiS=%g, dPhi=%g\n", dPhiS, dPhi);
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
        return 0;
    }

    double zoom(double lo, double philo, double dPhilo,
                      double hi, double phihi, double dPhihi,
                      double phiS, double dPhiS, double c1, double c2,
                      double alpha0, double phi0, double dPhi0,
                      double alpha1, double phi1, double dPhi1) {
        int j = 0;
        double alpha;
        double e = 1e-6;
        DMESSAGE("Zoom start.");
        while(j < 30) {
            alpha = interpolate(lo, philo, dPhilo, hi, phihi, dPhihi);
            DERROR("Interp: lo=%g, philo=%g, dlo=%g,\n \
                                      hi=%g, phihi=%g, dhi=%g, alpha=%g\n",
                               lo, philo, dPhilo, hi, phihi, dPhihi, alpha);
            assert(std::isfinite(alpha));
            computePhiDPhi(alpha);
            if(phi > phiS + c1*alpha*dPhiS || (phi >= philo)) {
                hi = alpha;
                phihi = phi;
                dPhihi = dPhi;
            } else {
                if(std::abs(dPhi) <= -c2*dPhiS) {
                    return alpha;
                }
                if(dPhi*(hi - lo) >= 0) {
                    hi = lo;
                    phihi = philo;
                    dPhihi = dPhilo;
                }
                lo = alpha;
                philo = phi;
                dPhilo = dPhi;
            }
            if(std::abs(alpha - alpha1) < e) {
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

    static double interpolate(double a, double fa, double dfa,
                             double b, double fb, double dfb) {
        double c = 5e-2;
        double d = b - a;
        double s = Math::signum(d);
        double d1 = dfa + dfb - 3 * (fa  - fb)/(a-b);
        double k = d1*d1 - dfa*dfb;
        double d2 = s * std::sqrt(k);
        double x= b - d * ((dfb + d2 - d1)/(dfb - dfa + 2*d2));
        if(std::abs(a-x) < c*d*s) {
            DERROR("Halving Interval: a=%g x=%g b=%g", a, x, b);
            x = a + 2*c*d;
        }
        if(std::abs(b-x) < c*d*s) {
            DERROR("Halving Interval: a=%g x=%g b=%g", a, x, b);
            x = b + 2*c*d;
        }
        assertTrue("Interpolated alpha is not in interval.",
                             x <= std::max(a, b) && x >= std::min(a, b));
        return x;
    }

    static double choose(double alpha1, double phi, double dphi,
                               double alphaMax){
            // TODO: make it smart
        return std::min(2*alpha1, alphaMax);
    }

    void computePhiDPhi(double alpha)  {
        xAlpha = alpha*dn + x0;
        phi = computeOptimizeGradient(xAlpha, g);
        // dPhi = dot(dn, g)/normP2(dn);
        computeDPhi(g, dn);
    }

    // void scanLineToFile(double alphaMax, int n, std::string filename) {
    //     assert(alphaMax >0);
    //     try {
    //         Path path =  Paths.get("build/test-results/physics/ScanLine");
    //         if(Files.notExists(path)){
    //                 Files.createDirectories(path);
    //         }
    //         path = path.resolve(filename);
    //         BufferedWriter writer = Files.newBufferedWriter(path);
    //         writer.write(String.format("#alpha phi dPhi1 dPhi2\n"));
    //         double phi1 = phi;
    //         double phiMin = phi;
    //         double dPhiMin = dPhi;
    //         double alphaMin = 0;
    //         double step = alphaMax/n;
    //         computePhiDPhi(-alphaMax - step);
    //         for(double alpha=-alphaMax; alpha<alphaMax; alpha+=step) {
    //             phi1 = phi;
    //             computePhiDPhi(alpha);
    //             if(phiMin > phi) {
    //                 dPhiMin = dPhi;
    //                 phiMin = phi;
    //                 alphaMin = alpha;
    //             }
    //             writer.write(String.format("% 8g % 12.8f % 12.8f % 12.8f\n",
    //                          alpha, phi, dPhi, (phi - phi1)/step));
    //         }
    //         computePhiDPhi(0);
    //         DERROR("Line Scan min value: alphaMin=%4.4f,"+
    //                              " phiMin=%g, dPhiMin=%g\n", alphaMin, phiMin, dPhiMin);
    //         // DMESSAGE("Origin and direction" + x0 + dn);
    //         writer.close();
    //     } catch(IOException e) {
    //         throw new RuntimeException("Could not write data");
    //     }
    // }

    void computeForwardEulerStep(Eigen::ArrayXd& x2,Eigen::ArrayXd& x1,Eigen::ArrayXd& x0,
                                 const Eigen::ArrayXd& fExt) {
        x2 = x1;
        x1 = x0;
        x0 = 2*x1-x2;
        temp1 = fExt*MI;
        x0 = dt*dt*temp1 + x0;
        temp1 =  (x1 - x2)/dt;
        DERROR("||v||=%g\n", sqrt((temp1*temp1).sum()));
    }

    void implicitEulerStep() {
        double dG = 1e33;
        double phiOld = std::numeric_limits<double>::max();


        DERROR("Newton iteration start. dG=%g\n", dG);
        // std::cout << "x2=" <<  x2 << std::endl;
        // step forward and set the initial guess
        computeForwardEulerStep(x2, x1, x0, fExt);
        xHat = x0;

        iNewton=0;
        while(iNewton <= 20) {
            dG = newtonStep();
            if(dG < newtonAccuracy) {
                DERROR("Newton iteration stopped: i=%d, dG=%g\n",
                 iNewton, dG);
                return;
            }
            {
                if (phi >= phiOld*1.2) {
                     DERROR("Energy minimization did not converge!\n phi=%g, phiOld=%g\n",
                                      phi, phiOld);
                    assert(false);
                    break;
                }
            }
            phiOld = phi;
            iNewton++;
            DERROR("Newton iteration i=%d, dG=%g\n", iNewton,dG);
        }
        DERROR("Energy minimization did not stop after 40 iterations! \
                           dE=%g, dE'=%g\n\n", dG, dG);
        assert(false);
    }

};
