#include <Eigen/Dense>

double squarePointLineSegmentDistance(double px, double py,
                                            double p0x, double p0y,
                                            double p1x, double p1y);

bool pointInTriangle(double px, double py, double p0x, double p0y,
                     double p1x, double p1y, double p2x, double p2y);

void addCollisionPenaltyForce(
    unsigned int iPoint, unsigned int i, unsigned int j,
    const Eigen::ArrayXd& x, Eigen::ArrayXd& dest);

void addCollisionPenaltyForceDifferential(
    unsigned int iPoint, unsigned int i, unsigned int j,
    const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest);
