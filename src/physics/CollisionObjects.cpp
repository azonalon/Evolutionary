#include "physics/CollisionObjects.hpp"
#include <cmath>
#include <iostream>
using namespace Eigen;
using namespace std;
static double a = 0.01;
static double mu = 1;

inline double sign(double x) { return x < 0 ? -1 : 1; }
inline double clampZero(double a) { return a < 0 ? 0 : a; }

const BoundingBox& Rectangle::boundingBoxUntransformed() const {
    // Eigen::Vector2d a(box.x0, box.y0);
    // Eigen::Vector2d b(box.x1, box.y1);
    // a = aux * a;
    // b = aux * b;
    // return BoundingBox{a.x(), a.y(), b.x(), b.y()};
    return box;
};

/**
 * 
 * 
 * */
double Rectangle::computePenaltyGradient(const double* pp, double* vv,
                                         double dtbym, double* gp) const {
  // const double& px = pp[0], py=pp[1];
  Eigen::Map<const Eigen::Vector2d> pur(pp);
  Eigen::Map<Eigen::Vector2d> vur(vv);
  Eigen::Map<Eigen::Vector2d> gur(gp);
  Eigen::Vector2d p = aux * pur;
  Eigen::Vector2d v = aux.linear() * vur;
  Eigen::Vector2d g = aux.linear() * gur;
  // std::cout << aux << std::endl;
  // std::cout << aux.linear() << std::endl;
  double E=0;

  // double dx = w - abs(p.x()), dy = h - abs(p.y());
  // if (dx >= 0 && dx < dy) {
  //   g.x() -= a * 2 * dx * sign(p.x());
  //   // compute friction
  //   double c = 0;//1 - dtbym * mu * 2 * a * dx / abs(v.x());
  //   v.y() *= clampZero(c);
  //   E = dx * dx * a;
  // } else if (dy >= 0 && dy < dx) {
  //   g.y() -= a * 2 * dy * sign(p.y());
  //   double c = 0;//1 - dtbym * mu * 2 * a * dy / abs(v.y());
  //   v.x() *= clampZero(c);
  //   E = dy * dy * a;
  // }
  double dx = w - p.x(), dy = h - p.y();
  if (dx >= 0 && dx < w && dx < dy) {
    g.x() -= a * 2 * dx * sign(p.x());
    // compute friction
    double c = 0;//1 - dtbym * mu * 2 * a * dx / abs(v.x());
    v.y() *= clampZero(c);
    E = dx * dx * a;
  } else if (dy >= 0 && dy < h && dy < dx) {
    g.y() -= a * 2 * dy * sign(p.y());
    double c = 0;//1 - dtbym * mu * 2 * a * dy / abs(v.y());
    v.x() *= clampZero(c);
    E = dy * dy * a;
  }

  vur = aux.linear().inverse() * v;
  gur = aux.linear().inverse() * g;
  return E;
};

void Rectangle::computePenaltyGradientDifferential(const double* pp,
                                                   const double* dpp,
                                                   double* dgg) const {
  Eigen::Map<const Eigen::Vector2d> pur(pp);
  Eigen::Map<const Eigen::Vector2d> dpur(dpp);
  Eigen::Map<Eigen::Vector2d> dgur(dgg);

  Eigen::Vector2d p = aux * pur;
  Eigen::Vector2d dp = aux.linear() * dpur;
  Eigen::Vector2d dg = aux.linear() * dgur;

  // double dx = w - abs(p.x()), dy = h - abs(p.y());
  // if (dx > 0 && dx < dy) {
  //   dg.x() += a * 2 * dp.x();
  // } else if (dy > 0 && dy < dx) {
  //   dg.y() += a * 2 * dp.y();
  // }
  double dx = w - p.x(), dy = h - p.y();
  if (dx > 0 && dx < w && dx < dy) {
    dg.x() += a * 2 * dp.x();
  } else if (dy > 0 && dy < h && dy < dx) {
    dg.y() += a * 2 * dp.y();
  }
  dgur = aux.linear().inverse() * dg;
}
