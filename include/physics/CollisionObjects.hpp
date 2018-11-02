#include <Eigen/Dense>
#pragma once

struct BoundingBox {
  double x0, y0, x1, y1;
  inline bool contains(double x, double y) const {
    return x > x0 && y > y0 && x < x1 && y < y1;
  };
};

class CollisionObject {
 public:
  BoundingBox boxTransform;
  // computes force into f, friction into fr, all at position x
  virtual const BoundingBox& boundingBoxUntransformed() const = 0;
  const BoundingBox& boundingBox() {
    return boxTransform;
  };
  virtual double computePenaltyGradient(const double* x, double* v,
                                        double dtbym, double* f) const = 0;
  virtual void computePenaltyGradientDifferential(const double* x,
                                                  const double* dx,
                                                  double* df) const = 0;
  Eigen::Affine2d aux = Eigen::Affine2d::Identity();
  void rotate(double angle) {
    aux = aux.prerotate(-angle);
    updateBoundingBox();
  }
  void translate(double x, double y) {
    aux = aux.translate(Eigen::Vector2d(-x, -y));
    updateBoundingBox();
  }
  void updateBoundingBox() {
    auto bb = boundingBoxUntransformed();
    Eigen::Vector2d a(bb.x0, bb.y0), b(bb.x0, bb.y1), c(bb.x1, bb.y0),
        d(bb.x1, bb.y1);
    a = aux.inverse() * a;
    b = aux.inverse() * b;
    c = aux.inverse() * c;
    d = aux.inverse() * d;
    boxTransform.y1 = std::max({a.y(), b.y(), c.y(), d.y()});
    boxTransform.x1 = std::max({a.x(), b.x(), c.x(), d.x()});
    boxTransform.y0 = std::min({a.y(), b.y(), c.y(), d.y()});
    boxTransform.x0 = std::min({a.x(), b.x(), c.x(), d.x()});
  }
};

class Rectangle : public CollisionObject {
 public:
  Rectangle(double w, double h) : w(w), h(h), box{-w, -h, w, h} {
    aux = Eigen::Affine2d::Identity();
    boxTransform = box;
  };
  double w, h;
  BoundingBox box;
  const BoundingBox& boundingBoxUntransformed() const override;
  double computePenaltyGradient(const double*, double*, double dtbym,
                                double*) const override;
  void computePenaltyGradientDifferential(const double*, const double*,
                                          double*) const override;
};
