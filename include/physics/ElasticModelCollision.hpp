#include <Eigen/Dense>

extern double muSelfFriction;
extern double strengthSelfCollision;

double squarePointLineSegmentDistance(double px, double py,
                                            double p0x, double p0y,
                                            double p1x, double p1y);

bool pointInTriangle(double px, double py, double p0x, double p0y,
                     double p1x, double p1y, double p2x, double p2y);

void addSelfCollisionPenaltyForce(
    std::array<unsigned, 3> triplet,
    const VectorD& x, VectorD& dest);

void addSelfCollisionPenaltyForceDifferential(
    std::array<unsigned, 3> triplet,
    const VectorD& x, const VectorD& dx, VectorD& dest);
