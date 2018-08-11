#include <Eigen/Dense>

double squarePointLineSegmentDistance(double px, double py,
                                            double p0x, double p0y,
                                            double p1x, double p1y);

bool pointInTriangle(double px, double py, double p0x, double p0y,
                     double p1x, double p1y, double p2x, double p2y);

void addSelfCollisionPenaltyForce(
    std::array<unsigned, 3> triplet,
    const Eigen::ArrayXd& x, Eigen::ArrayXd& dest);

void addSelfCollisionPenaltyForceDifferential(
    std::array<unsigned, 3> triplet,
    const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest);
