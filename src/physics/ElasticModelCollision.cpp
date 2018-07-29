#include "physics/ElasticModel.hpp"
static double c = 10000;

double squarePointLineSegmentDistance(double px, double py,
                                            double p0x, double p0y,
                                            double p1x, double p1y) {
    double dX10 = p1x - p0x;
    double dY10 = p1y - p0y;
    double dX0  = px  - p0x;
    double dY0  = py  - p0y;
    double s = (dX10*dY0 - dX0*dY10);
    double Lsquare = dX10*dX10 + dY10*dY10;
    return s*s/Lsquare;
};

bool pointInTriangle(double px, double py, double p0x, double p0y,
                     double p1x, double p1y, double p2x, double p2y) {
    double dX2 = px-p2x;
    double dY2 = py-p2y;
    double dX12 = p1x-p2x;
    double dY12 = p1y-p2y;
    double dX02 = p0x-p2x;
    double dY02 = p0y-p2y;

    double D =  dY12*dX02  - dX12*dY02;
    double s =  dY12*dX2   - dX12*dY2;
    double t = -dY02*dX2   + dX02*dY2;
    if (D<0) {
        return s<=0 && t<=0 && s+t>=D;
    }
    return s>=0 && t>=0 && s+t<=D;
}

// adds force resulting from point iPoint inside surface (surface is a line from
// vertex i to j)
double addCollisionPenaltyForce(
    unsigned int iPoint, unsigned int i, unsigned int j,
    const Eigen::ArrayXd& x, Eigen::ArrayXd& dest) {

    double px = x(iPoint), py = x(iPoint+1);
    double p0x = x(i), p0y = x(i+1);
    double p1x = x(j), p1y = x(j+1);
    double dX1 = px-p1x;
    double dY1 = py-p1y;
    double dX0 = px-p0x;
    double dY0 = py-p0y;
    double dX10 = p1x-p0x;
    double dY10 = p1y-p0y;

    double s = -dX10*dY0  + dX0*dY10;
    double s2 = s*s;

    double L2 = dX10*dX10 + dY10*dY10;
    double L4 = L2*L2;
    double L6 = L4*L2;


    double eps = 1e-15;
    double U2 = s2/L2 + eps;
    double U = sqrt(U2);
    double phi= s2/L2*U;

    // double q1 = s2 + 2*L2 * U2;
    // double q2 = q1 + 6*s2;
    // double q3 = 6*q1*s + 4 *s2*s;

    double kf = s2*s/(L6*U) + 2*s*U/L4;

    dest[iPoint + 0] -= c * kf*( dY10 *L2         );
    dest[iPoint + 1] -= c * kf*(-dX10 *L2         );
    dest[i + 0]      -= c * kf*( dY1  *L2 + dX10*s);
    dest[i + 1]      -= c * kf*(-dX1  *L2 + dY10*s);
    dest[j + 0]      -= c * kf*(-dY0  *L2 - dX10*s);
    dest[j + 1]      -= c * kf*( dX0  *L2 - dY10*s);

    return c*phi;
}

void addCollisionPenaltyForceDifferential(
    unsigned int iPoint, unsigned int i, unsigned int j,
    const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest) {
    double px = x(iPoint), py = x(iPoint+1);
    double p0x = x(i), p0y = x(i+1);
    double p1x = x(j), p1y = x(j+1);
    double dX1 = px-p1x;
    double dY1 = py-p1y;
    double dX0 = px-p0x;
    double dY0 = py-p0y;
    double dX10 = p1x-p0x;
    double dY10 = p1y-p0y;
    double L2 = dX10*dX10 + dY10*dY10;
    double L4 = L2*L2;
    double L6 = L4*L2;
    double L10 = L4*L6;
    double s = -dX10*dY0  + dX0*dY10;
    double s2 = s*s;
    double eps = 1e-15;
    double U2 = s2/L2 + eps;
    double U = sqrt(U2);
    double U3 = U2*U;
    double U4 = U2*U2;

    double q1 = s2 + 2*L2 * U2;
    double q2 = q1 + 6*s2;
    double q3 = 6*q1*s + 4 *s2*s;

    double kf = s2*s/(L6*U) + 2*s*U/L4;

    Eigen::Matrix<double, 6, 1> f;
    Eigen::Matrix<double, 6, 1> kfd;
    Eigen::Matrix<double, 6, 6> fd;

    f <<
      dY10 *L2,
     -dX10 *L2,
      dY1  *L2 + dX10*s,
     -dX1  *L2 + dY10*s,
     -dY0  *L2 - dX10*s,
      dX0  *L2 - dY10*s;


    kfd <<
        -(dY10*L2*q1*s2) + dY10*L4*q2*U2,
         dX10*L2*q1*s2 - dX10*L4*q2*U2,
        -(q1*(dY1*L2 + dX10*s)*s2) + L2*(dY1*L2*q2 + dX10*q3)*U2 - 4*dX10*L4*s*U4,
        q1*(dX1*L2 - dY10*s)*s2 + L2*(-(dX1*L2*q2) + dY10*q3)*U2 - 4*dY10*L4*s*U4,
        q1*(dY0*L2 + dX10*s)*s2 - L2*(dY0*L2*q2 + dX10*q3)*U2 + 4*dX10*L4*s*U4,
        q1*(-(dX0*L2) + dY10*s)*s2 + L2*(dX0*L2*q2 - 2*dY10*s*(3*q1 + 2*s2))*U2 + 4*dY10*L4*s*U4;

    fd <<
        0, 0, -2*dX10*dY10, -dX10 * dX10 - 3*dY10 * dY10, 2*dX10*dY10, dX10 * dX10 + 3*dY10 * dY10,
        0, 0, 3*dX10 * dX10 + dY10 * dY10, 2*dX10*dY10, -3*dX10 * dX10 - dY10 * dY10, -2*dX10*dY10,
        dX10*dY10, dY10 * dY10, dX10*dY0 - dX10*dY1 - dX0*dY10, -(dX1*dX10) - 2*dY1*dY10, -2*dX10*dY0 + 2*dX10*dY1 + dX0*dY10, dX0*dX10 + 2*dY1*dY10 - dX10 * dX10 - dY10 * dY10,
        -dX10 * dX10, -(dX10*dY10), 2*dX1*dX10 + dY1*dY10, dX10*dY0 - dX0*dY10 + dX1*dY10, -2*dX1*dX10 - dY0*dY10 + L2, -(dX10*dY0) + 2*dX0*dY10 - 2*dX1*dY10,
        -(dX10*dY10), -dY10 * dY10, dX10*dY0 - dX10*dY1 + dX0*dY10, dX1*dX10 + 2*dY0*dY10 + L2, -(dX0*dY10), -(dX0*dX10) - 2*dY0*dY10,
        dX10 * dX10, dX10*dY10, -2*dX0*dX10 - dY1*dY10 - dX10 * dX10 - dY10 * dY10, -(dX10*dY0) - dX0*dY10 + dX1*dY10, 2*dX0*dX10 + dY0*dY10, dX10*dY0;

    Eigen::Matrix<double, 6, 6> h = 1/(L10*U3) * f*kfd.transpose() + kf * fd;
    Eigen::Matrix<double, 6, 1> dxtmp;
    dxtmp << dx(iPoint), dx(iPoint+1), dx(i), dx(i+1), dx(j), dx(j+1);
    Eigen::Matrix<double, 6, 1> df = h*dxtmp;
    dest(iPoint + 0) -= c*df(0);
    dest(iPoint + 1) -= c*df(1);
    dest(i + 0)      -= c*df(2);
    dest(i + 1)      -= c*df(3);
    dest(j + 0)      -= c*df(4);
    dest(j + 1)      -= c*df(5);
}


double ElasticModel::computeCollisionPenaltyForce(
    unsigned int iPoint, unsigned int i, unsigned int j, unsigned int k,
    const Eigen::ArrayXd& x, Eigen::ArrayXd& dest)
{
    // std::cout << "point is in triangle? " <<  pointInTriangle(
            // x(iPoint+0), x(iPoint+1),
            // x(i+0), x(i+1),
            // x(j+0), x(j+1),
            // x(k+0), x(k+1)) << std::endl;
    // std::cout << "x=" << x.transpose() << std::endl;
    // printf("iPoint=%d, i=%d, j=%d, k=%d\n", iPoint, i, j, k);
    if(iPoint != i && iPoint !=j && iPoint != k
        && pointInTriangle(
            x(iPoint+0), x(iPoint+1),
            x(i+0), x(i+1),
            x(j+0), x(j+1),
            x(k+0), x(k+1))) {
        double d1 = squarePointLineSegmentDistance(
            x(iPoint+0), x(iPoint+1),
            x(i+0), x(i+1),
            x(j+0), x(j+1));
        double d2 = squarePointLineSegmentDistance(
            x(iPoint+0), x(iPoint+1),
            x(i+0), x(i+1),
            x(k+0), x(k+1));
        double d3 = squarePointLineSegmentDistance(
            x(iPoint+0), x(iPoint+1),
            x(j+0), x(j+1),
            x(k+0), x(k+1));
        if(d1 < d2 && d1 < d3) {
            return addCollisionPenaltyForce(iPoint, i, j, x, dest);
        } else if(d2 < d3 && d2 < d1) {
            return addCollisionPenaltyForce(iPoint, i, k, x, dest);
        }
        else {
            return addCollisionPenaltyForce(iPoint, j, k, x, dest);
        }
    }
    return 0.0;
};

void ElasticModel::computeCollisionPenaltyForceDifferential(
    unsigned int iPoint, unsigned int i, unsigned int j, unsigned int k,
    const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest)
{
    if(iPoint != i && iPoint !=j && iPoint != k
        && pointInTriangle(
            x(iPoint+0), x(iPoint+1),
            x(i+0), x(i+1),
            x(j+0), x(j+1),
            x(k+0), x(k+1))) {
        double d1 = squarePointLineSegmentDistance(
            x(iPoint+0), x(iPoint+1),
            x(i+0), x(i+1),
            x(j+0), x(j+1));
        double d2 = squarePointLineSegmentDistance(
            x(iPoint+0), x(iPoint+1),
            x(i+0), x(i+1),
            x(k+0), x(k+1));
        double d3 = squarePointLineSegmentDistance(
            x(iPoint+0), x(iPoint+1),
            x(j+0), x(j+1),
            x(k+0), x(k+1));
        if(d1 < d2 && d1 < d3) {
            addCollisionPenaltyForceDifferential(iPoint, i, j, x, dx, dest);
        } else if(d2 < d3 && d2 < d1) {
            addCollisionPenaltyForce(iPoint, i, k, x, dest);
        }
        else {
            addCollisionPenaltyForceDifferential(iPoint, j, k, x, dx, dest);
        }
    }
};
