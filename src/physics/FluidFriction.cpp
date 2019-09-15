#include "physics/ElasticModel.hpp"

double ElasticModel::computeFluidFrictionGradient(const Eigen::ArrayXd& x,
                                                  Eigen::ArrayXd& dest) {
  //TODO handle last surface vertex and optimize redundancy for swapping j and i
  double E = 0;
  double kF = kFluid/dt/dt/dt;
  for (auto& s : surfaces) {
    unsigned ii=s.size()-1;
    for (unsigned jj=0; jj<s.size(); jj++) {
      unsigned i = s[ii];
      unsigned j = s[jj];
      double vx =   (x[2*i] -     x1[2*i]);
      double vy =   (x[2*i + 1] - x1[2*i+1]);
      double px =   (x1[2*i + 0] - x1[2*j+0]);
      double py =   (x1[2*i + 1] - x1[2*j+1]);
      double tmp1 = (-vx*py + vy*px);
      double tmp2 = (vx*px + vy*py);
      double tmp1sq = tmp1*tmp1;
      double tmp2sq = tmp2*tmp2;
      if (tmp2 > 0) {
        E += (tmp1sq*tmp1sq + tmp2sq*tmp2sq)*kF / 4.0;
        dest[2 * i + 0]  +=  kF * ( - py * tmp1sq * tmp1 + px * tmp2sq * tmp2);
        dest[2 * i + 1]  +=  kF * ( + px * tmp1sq * tmp1 + py * tmp2sq * tmp2);
      }
      ii = jj;
    }
  }
  return E;
}

void ElasticModel::computeFluidFrictionGradientDifferential(
    const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest) {
  double kF = kFluid/dt/dt/dt;
  for (auto& s : surfaces) {
    unsigned ii=s.size()-1;
    for (unsigned jj=0; jj<s.size(); jj++) {
      unsigned i = s[ii];
      unsigned j = s[jj];
      double vx =  ( x[2*i] -     x1[2*i]);
      double vy =  ( x[2*i+1] - x1[2*i+1]);
      double px =  (x1[2*i+0] - x1[2*j+0]);
      double py =  (x1[2*i+1] - x1[2*j+1]);
      double tmp1= (-vx*py + vy*px);
      double tmp2 = (vx*px + vy*py);
      double tmp1sq = tmp1*tmp1;
      double tmp2sq = tmp2*tmp2;
      if (tmp2 > 0) {
        dest[2 * i + 0] +=  3*kF* (dx[2 * i + 0] * (py*py*tmp1sq + px*px*tmp2sq) + 
                                   dx[2 * i + 1] * px*py* (-tmp1sq + tmp2sq)    );
        dest[2 * i + 1] +=  3*kF* (dx[2 * i + 0] * px*py* (-tmp1sq + tmp2sq) + 
                                   dx[2 * i + 1] * (px*px*tmp1sq + py*py*tmp2sq) );
      }
      ii = jj;
    }
  }
}