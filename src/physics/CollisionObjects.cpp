#include "physics/CollisionObjects.hpp"
#include <cmath>
using namespace Eigen;
using namespace std;
static double a = 1000;
static double mu = 100000;

inline double sign(double x) {
    return x < 0 ? -1: 1;
}
inline double clampZero(double a) {
    return a < 0 ? 0:a;
}

const BoundingBox& Rectangle::boundingBox() const {
    return box;
};

// double Rectangle::computeDistanceGradient(const double* pp,
//                                                 double* gp) const {
//     const double& px = pp[0], py=pp[1];
//     double dx = w - abs(px), dy = h - abs(py);
//     if(dx < dy) {
//         *(gp+0) += sign(px);
//         return dx;
//     } else {
//         *(gp+1) += sign(py);
//         return dy;
//     }
// };
//
// void Rectangle::computeDistanceGradientDifferential(const double* x,
//                                                     const double* dx,
//                                                     double* dg) const {
//     // is zero
//     return;
// }



double Rectangle::computePenaltyForce(const double* pp,
                                        double* v, double dtbym,
                                        double* gp) const {
    const double& px = pp[0], py=pp[1];
    double dx = w - abs(px), dy = h - abs(py);
    if(dx > 0 && dx < dy) {
        *(gp+0) += a*2*dx*sign(px);
        // compute friction
        double c = 1 - dtbym*mu*2*a*dx / ( abs(*(v+1)));
        *(v+1) *= 0*clampZero(c);
        return dx*dx*a;
    } else if(dy > 0 && dy < dx){
        *(gp+1) += a*2*dy*sign(py);
        double c = 1 - dtbym*mu*2*a*dy / ( abs(*(v+1)));
        *(v+0) *= 0*clampZero(c);
        return dy*dy*a;
    }
    return 0;
};

void Rectangle::computePenaltyForceDifferential(const double* p,
                                                    const double* dp,
                                                    double* dg) const {
    const double& px = p[0], py=p[1];
    double dx = w - abs(px), dy = h - abs(py);
    if(dx > 0 && dx < dy) {
        *(dg + 0) -= a* 2 * *(dp+0);
    } else if(dy > 0 && dy<dx){
        *(dg + 1) -= a* 2 * *(dp+1);
    }
    return;
}
