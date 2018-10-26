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
    virtual const BoundingBox& boundingBox() const =0;
    // virtual double computeDistanceGradient(const double*, double*) const  =0;
    // virtual void computeDistanceGradientDifferential(const double* x, const double* dx, double* df) const=0;
    // computes force into f, friction into fr, all at position x
    virtual double computePenaltyGradient(const double* x, double* v, double dtbym, double* f) const  =0;
    virtual void computePenaltyGradientDifferential(const double* x, const double* dx, double* df) const=0;
};

class Rectangle: public CollisionObject{
public:
    Rectangle(double w, double h): w(w), h(h), box{-w, -h, w, h} { };
    double w, h;
    BoundingBox box;
    const BoundingBox& boundingBox() const override;
    // double computeDistanceGradient(const double* , double*) const override;
    // void computeDistanceGradientDifferential(const double*, const double*, double*) const override;
    double computePenaltyGradient(const double* , double*, double dtbym, double*) const override;
    void computePenaltyGradientDifferential(const double*, const double*, double*) const override;
};
